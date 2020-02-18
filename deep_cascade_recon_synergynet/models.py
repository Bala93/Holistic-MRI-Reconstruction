import torch
from torch import nn
from torch.nn import functional as F
import os 
import numpy as np
import itertools
from collections import OrderedDict
import math

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

class DenseConnectionLayer(nn.Module):
    def __init__(self, in_chans, k):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(k):
            self.layers+=[nn.Conv2d(in_chans, 1, kernel_size=3, padding=1)]

    def forward(self,x,k):
        #print(x.shape)
        out = []
        for i in range(k):
            xout = self.layers[i](x[i])
            #print(xout.shape)
            out.append(xout)
        
        return torch.stack(out)


class DenseBlock(nn.Module):

    def __init__(self, n_ch, nd=5, nf=32, ks=3, dilation=1, bn=False, nl='lrelu', conv_dim=2, n_out=1):
        super().__init__()

        # convolution dimension (2D or 3D)
        if conv_dim == 2:
            conv = nn.Conv2d
        else:
            conv = nn.Conv3d
        self.nd = nd 

        # output dim: If None, it is assumed to be the same as n_ch
        #if not n_out:
        #    n_out = n_ch

        # dilated convolution
        pad_conv = 1
        if dilation > 1:
            # in = floor(in + 2*pad - dilation * (ks-1) - 1)/stride + 1)
            # pad = dilation
            pad_dilconv = dilation
        else:
            pad_dilconv = pad_conv

        def conv_i(n_i):
            return conv(n_i, nf, ks, stride=1, padding=pad_dilconv, dilation=dilation, bias=True)

        conv_1 = conv(n_ch, nf, ks, stride=1, padding=pad_conv, bias=True)
        conv_n = conv(nf+nd, n_out, ks, stride=1, padding=pad_conv, bias=True)
        #print(conv_1) 
        #print(conv_n) 
        # relu
        nll = relu if nl == 'relu' else lrelu
        self.denseConnlayers = nn.ModuleList([])
        self.layers = nn.ModuleList([conv_1, nll()])
        for i in range(nd-1):
            #if bn:
            #    layers.append(nn.BatchNorm2d(nf))
            #print(DenseConnectionLayer(nf,i+1))
            #print("############################")
            self.denseConnlayers += [DenseConnectionLayer(nf,i+1)]
            self.layers += [conv_i(nf+i+1), nll()]

        self.denseConnlayers += [DenseConnectionLayer(nf,nd)]
        self.layers += [conv_n]

    def forward(self,x):
        #print("x entering: ", x.shape) #[1, 1, 150, 150]
        flist=[]
        xtemp = self.layers[0](x) # first conv that takes image input, gives out 32 feature maps
        xtemp = self.layers[1](xtemp) # 1st ReLU
        #print("x temp after first conv+ReLU: ", xtemp.shape)#[1, 32, 150, 150])
        x = x.unsqueeze(0)
        flist.append(xtemp)
        for i in range(1,self.nd):
            featureList = torch.stack(flist)
            #print("featureList shape: ", featureList.shape) #[1, 1, 32, 150, 150]
            x1 = self.denseConnlayers[i-1](featureList,i) #takes i+1 list each having 32 feature maps gives i+1 feature maps as output 
            #print("x1 output of denseconn shape: ", x1.shape, "i value: ",i) #[1, 1, 1, 150, 150]
            xtemp = xtemp.unsqueeze(0)
            x1 = x1.permute(2,1,0,3,4)
            #print("x temp shape: ", xtemp.shape, "x1.shape: ", x1.shape)
            #x shape:  torch.Size([1, 1, 1, 150, 150]) x1.shape:  torch.Size([1, 1, 1, 150, 150])
            x3 = torch.cat([xtemp,x1],dim=2) 
            x3 = x3.squeeze(0)
            #print("x3 shape: ", x3.shape)
            #print("self.layers[2*1]",self.layers[2*i], " 2*i: ",2*i)
            x4 = self.layers[2*i](x3) # takes ith conv layer 32+i feature maps
            x4 = self.layers[2*i+1](x4) # takes ith ReLU layer
            #print("x4 shape: ", x4.shape)# [1, 32, 150, 150]
            xtemp=x4
            #print("xtemp shape: ", xtemp.shape)# [1, 32, 150, 150]
            flist.append(xtemp)
            #print("flist[i] shape: ",flist[i].shape)
            #print("flist len: ",len(flist))
          
        featureList = torch.stack(flist)
        #print("final featureList shape: ", featureList.shape) #[1, 1, 32, 150, 150]
        x1 = self.denseConnlayers[self.nd-1](featureList,self.nd) #takes i+1 list each having 32 feature maps gives i+1 feature maps as output 
        #print("final x1 output of denseconn shape: ", x1.shape, "nd value: ",nd) #[1, 1, 1, 150, 150]
        xtemp = xtemp.unsqueeze(0)
        #print("final xtemp shape after unsqueeze: ", xtemp.shape) #[1, 1, 1, 150, 150]
        x1 = x1.permute(2,1,0,3,4)
        #print("final x temp shape: ", xtemp.shape, "x1.shape: ", x1.shape)
        x3 = torch.cat([xtemp,x1],dim=2) 
        x3 = x3.squeeze(0)
        #print("final x3 shape: ", x3.shape)
        #print("final self.layers[2*1]",self.layers[2*nd], " 2*i: ",2*nd)
        out = self.layers[2*self.nd](x3) # takes ith conv layer 32+i feature maps
        return out

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
        self.avgpoollayer = nn.AvgPool2d((w, h))
        self.conv1 = nn.Conv2d(in_channel, int(in_channel/r), kernel_size=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(int(in_channel/r), in_channel, kernel_size=1)
        self.sigmoid1 = nn.Sigmoid()
        



    def forward(self, x):
#        print("x entering CSE block: ", x.shape)
        s = self.layer(x)
#        print("s after layer in CSE block: ", s.shape)
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

class HyperDenseBlock2(nn.Module):

    def __init__(self, n_ch, nd=5, nf=32, ks=3, dilation=1, bn=False, nl='lrelu', conv_dim=2, n_out=1):
        super().__init__()


        def conv_i(n_i):
            return nn.Conv2d(n_i, nf, ks, stride=1, padding=1, dilation=dilation, bias=True)

        conv_1_top = nn.Conv2d(1, nf, ks, stride=1, padding=1, bias=True)
        conv_1_bottom = nn.Conv2d(1, nf, ks, stride=1, padding=1, bias=True)
        conv_n_top = nn.Conv2d(nf+(2*nd), n_out, ks, stride=1, padding=1, bias=True)
        conv_n_bottom = nn.Conv2d(nf+(2*nd), n_out, ks, stride=1, padding=1, bias=True)
        conv_n = nn.Conv2d(2, n_out, ks, stride=1, padding=1, bias=True)
        self.nd = nd
        nll = relu if nl == 'relu' else lrelu
        self.topdenseConnlayers = nn.ModuleList([])
        self.bottomdenseConnlayers = nn.ModuleList([])
        self.toplayers = nn.ModuleList([])
        self.toplayers += [nn.Sequential(conv_1_top, nll())]
        self.bottomlayers = nn.ModuleList([])
        self.bottomlayers += [nn.Sequential(conv_1_bottom, nll())]
        for i in range(nd-1):
            self.topdenseConnlayers += [DenseConnectionLayer(nf,i+1)]
            self.bottomdenseConnlayers += [DenseConnectionLayer(nf,i+1)]
            self.toplayers += [nn.Sequential(conv_i(nf+(2*(i+1))), nll())]
            self.bottomlayers += [nn.Sequential(conv_i(nf+(2*(i+1))), nll())]

        self.topdenseConnlayers += [DenseConnectionLayer(nf,nd)]
        self.bottomdenseConnlayers += [DenseConnectionLayer(nf,nd)]
        self.toplayers += [conv_n_top]
        self.bottomlayers += [conv_n_bottom]
        self.outlayer = conv_n

    def forward(self,xtop,xbottom):
        #print("x entering: ", x.shape) #[1, 1, 150, 150]
        topflist=[]
        bottomflist=[]
        xtoptemp = self.toplayers[0](xtop) # first conv that takes image input, gives out 32 feature maps
        xbottomtemp = self.bottomlayers[0](xbottom) # first conv that takes image input, gives out 32 feature maps
        topflist.append(xtoptemp)
        bottomflist.append(xbottomtemp)
        for i in range(1,self.nd):
            topfeatureList = torch.stack(topflist)
            bottomfeatureList = torch.stack(bottomflist)
            #print("topfeaturelist shape: ", topfeatureList.shape, "bottomfeaturelist.shape: ", bottomfeatureList.shape)
            x1top = self.topdenseConnlayers[i-1](topfeatureList,i) #takes i+1 list each having 32 feature maps gives i+1 feature maps as output 
            x1bottom = self.bottomdenseConnlayers[i-1](bottomfeatureList,i) #takes i+1 list each having 32 feature maps gives i+1 feature maps as output 
            x1top = x1top.permute(2,1,0,3,4)
            x1bottom = x1bottom.permute(2,1,0,3,4)
            #print("x1toptemp shape: ", xtoptemp.shape, "x1top.shape: ", x1top.shape, "x1bottom.shape: ",x1bottom.shape)
            x3top = torch.cat([xtoptemp,x1top[0],x1bottom[0]],dim=1) 
            x3bottom = torch.cat([xbottomtemp,x1bottom[0],x1top[0]],dim=1)
            #print("x3top shape: ", x3top.shape, " x3bottom: ", x3bottom.shape) 
            x4top = self.toplayers[i](x3top) # takes ith conv layer 32+i feature maps
            x4bottom = self.bottomlayers[i](x3bottom) # takes ith conv layer 32+i feature maps
            #print("x4top shape: ", x4top.shape, " x4bottom: ", x4bottom.shape) 
            xtoptemp=x4top #store in temp for next iteration
            xbottomtemp=x4bottom
            topflist.append(xtoptemp)
            bottomflist.append(xbottomtemp)
          
        topfeatureList = torch.stack(topflist)
        bottomfeatureList = torch.stack(bottomflist)
        x1top = self.topdenseConnlayers[self.nd-1](topfeatureList,self.nd) #takes i+1 list each having 32 feature maps gives i+1 feature maps as output 
        x1bottom = self.bottomdenseConnlayers[self.nd-1](bottomfeatureList,self.nd) #takes i+1 list each having 32 feature maps gives i+1 feature maps as output 
        x1top = x1top.permute(2,1,0,3,4)
        x1bottom = x1bottom.permute(2,1,0,3,4)
        x3top = torch.cat([xtoptemp,x1top[0], x1bottom[0]],dim=1) 
        x3bottom = torch.cat([xbottomtemp,x1bottom[0], x1top[0]],dim=1) 
        outtop = self.toplayers[self.nd](x3top) # takes ith conv layer 32+i feature maps
        outbottom = self.bottomlayers[self.nd](x3bottom) # takes ith conv layer 32+i feature maps
        out = torch.cat([outtop,outbottom],dim=1)
        return self.outlayer(out)

############################################# Densenet and hyperdensener Implementation 2 ##################################################
class DenseLayer(nn.Module):
    
    def __init__(self,nc):
        super(DenseLayer,self).__init__()
        
        self.conv = nn.Conv2d(32,1,3,1,1)
        dense_connection = []
        self.nc = nc
        
        for ii in range(nc):
            dense_connection.append(self.conv)
        
        self.dense_connection = nn.ModuleList(dense_connection)
    
    def forward(self,x):
        
        y1 = []
        
        for ii in range(self.nc):
            y1.append(self.dense_connection[ii](x[ii]))
            
        y2 = torch.cat(y1,dim=1)
        
        return y2

class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet,self).__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(2,32,3,1,1),nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(32,32,3,1,1),nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(33,32,3,1,1),nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(34,32,3,1,1),nn.ReLU())
        self.conv5 = nn.Conv2d(35,1,3,1,1)
        
        self.dense = nn.ModuleList([
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5
        ])

        dense_layers = []

        for ii in range(1,4):
            dense_layers.append(DenseLayer(ii))

        self.dense_layers = nn.ModuleList(dense_layers)
        
    def forward(self,x):
#         print (x.shape)        
        x1 = self.dense[0](x)        
        dense_features = []
        dense_features.append(x1)        
#         print (x1.shape)        
        for ii in range(1,4):            
            x1 = self.dense[ii](x1)
#             print (x1.shape)
            dense_features.append(x1)
            y1_dc = self.dense_layers[ii-1](dense_features)
            x1 = torch.cat([y1_dc,x1],dim=1)
#             print (x1.shape)        
        x1 = self.dense[4](x1)        
        return x1


class HyperDenseNet(nn.Module):

    def __init__(self):
        super(HyperDenseNet,self).__init__()
        
        self.conv11 = nn.Sequential(nn.Conv2d(1,32,3,1,1),nn.ReLU())
        self.conv12 = nn.Sequential(nn.Conv2d(32,32,3,1,1),nn.ReLU())
        self.conv13 = nn.Sequential(nn.Conv2d(34,32,3,1,1),nn.ReLU())
        self.conv14 = nn.Sequential(nn.Conv2d(36,32,3,1,1),nn.ReLU())
        self.conv15 = nn.Sequential(nn.Conv2d(38,1,3,1,1),nn.ReLU())

        self.conv21 = nn.Sequential(nn.Conv2d(1,32,3,1,1),nn.ReLU())
        self.conv22 = nn.Sequential(nn.Conv2d(32,32,3,1,1),nn.ReLU())
        self.conv23 = nn.Sequential(nn.Conv2d(34,32,3,1,1),nn.ReLU())
        self.conv24 = nn.Sequential(nn.Conv2d(36,32,3,1,1),nn.ReLU())
        self.conv25 = nn.Sequential(nn.Conv2d(38,1,3,1,1),nn.ReLU())
        
        self.conv_final = nn.Conv2d(2,1,3,1,1)
        
        self.dense1 = nn.ModuleList([
            self.conv11,
            self.conv12,
            self.conv13,
            self.conv14,
            self.conv15
        ])

        
        self.dense2 = nn.ModuleList([  
            self.conv21,
            self.conv22,
            self.conv23,
            self.conv24,
            self.conv25
        ])
        

        dense_layers = {}

        for ii in range(1,4):
            dense_layers[str(ii)] = nn.ModuleList([DenseLayer(ii),DenseLayer(ii),DenseLayer(ii),DenseLayer(ii)]) #top row, bottom row, top to down, down to top

        self.dense_layers = nn.ModuleDict(dense_layers)
        
    def forward(self,x1,x2):
        
        x1 = self.dense1[0](x1)
        x2 = self.dense2[0](x2)


        dense1_features = []
        dense2_features = []
        
        dense1_features.append(x1)
        dense2_features.append(x2)
        
        for ii in range(1,4):
    
            x1 = self.dense1[ii](x1)
            x2 = self.dense2[ii](x2)
#             print (x1.shape)
#             print (x2.shape,y2_dc.shape,y12_dc.shape)

            dense1_features.append(x1)
            dense2_features.append(x2)
            
            #y1_dc = DenseLayer(ii)(self.dense1_features)
            #y2_dc = DenseLayer(ii)(self.dense2_features)
            y1_dc = self.dense_layers[str(ii)][0](dense1_features)
            y2_dc = self.dense_layers[str(ii)][1](dense2_features)
#             print (y1_dc.shape)
            
            #y21_dc = DenseLayer(ii)(self.dense2_features)
            #y12_dc = DenseLayer(ii)(self.dense1_features)
            y21_dc = self.dense_layers[str(ii)][2](dense2_features)
            y12_dc = self.dense_layers[str(ii)][3](dense1_features)
#             print (y21_dc.shape)
            
            x1 = torch.cat([x1,y1_dc,y21_dc],dim=1)
            x2 = torch.cat([x2,y1_dc,y12_dc],dim=1)
            
#             print (x1.shape)
#             print (x2.shape,y2_dc.shape,y12_dc.shape)    
    
        x1 = self.dense1[4](x1)
        x2 = self.dense2[4](x2)
        x_cat = torch.cat([x1,x2],dim=1)
        x  = self.conv_final(x_cat)

        return x

##############################################end of Densenet and hyperdensenet implementation 2 #############################################

class FastMRIUnetConvBlock(nn.Module):
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
        return self.layers(input)

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

        self.down_sample_layers = nn.ModuleList([FastMRIUnetConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [FastMRIUnetConvBlock(ch, ch * 2, drop_prob)]
            ch *= 2
        self.conv = FastMRIUnetConvBlock(ch, ch, drop_prob)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [FastMRIUnetConvBlock(ch * 2, ch // 2, drop_prob)]
            ch //= 2
        self.up_sample_layers += [FastMRIUnetConvBlock(ch * 2, ch, drop_prob)]
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

class SCSEUnetModel(nn.Module):
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

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob,attention=True,attention_type=attention_type,reduction=reduction)])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, drop_prob,attention=True,attention_type=attention_type,reduction=reduction)]
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
        self.us_mask = self.us_mask.float() 

    def forward(self,predicted_img,us_kspace):

        # us_kspace     = us_kspace[:,0,:,:]
        predicted_img = predicted_img[:,0,:,:]
        
        #kspace_predicted_img = torch.rfft(predicted_img,2,True,False).double()
        kspace_predicted_img = torch.rfft(predicted_img,2,True,False)
        #print (us_kspace.shape,predicted_img.shape,kspace_predicted_img.shape,self.us_mask.shape)
        # for kirby torch.Size([4, 1, 256, 256, 2]) torch.Size([4, 256, 256]) torch.Size([4, 256, 256, 2]) torch.Size([1, 256, 256, 1])
        us_kspace = us_kspace.permute(0,2,3,1)
        #print ("us_kspace: ",us_kspace.shape," predicted_img: ",predicted_img.shape," kspace_predicted_img: ",kspace_predicted_img.shape," mask: ",self.us_mask.shape)
        updated_kspace1  = self.us_mask * us_kspace 
        updated_kspace2  = (1 - self.us_mask) * kspace_predicted_img
        #print("updated_kspace1 shape: ",updated_kspace1.shape," updated_kspace2 shape: ",updated_kspace2.shape)
        #updated_kspace1 shape:  torch.Size([4, 1, 256, 256, 2])  updated_kspace2 shape:  torch.Size([4, 256, 256, 2])
        updated_kspace   = updated_kspace1 + updated_kspace2
        
        
        updated_img    = torch.ifft(updated_kspace,2,True) 
        
        #update_img_abs = torch.sqrt(updated_img[:,:,:,0]**2 + updated_img[:,:,:,1]**2)
        update_img_abs = updated_img[:,:,:,0]
        
        update_img_abs = update_img_abs.unsqueeze(1)
        
        return update_img_abs.float(), updated_kspace



def init_noise_(tensor, init):
    with torch.no_grad():
        return getattr(torch.nn.init, init)(tensor) if init else tensor.zero_()


def init_fourier_(tensor, norm='ortho'):
    """Initialise convolution weight with Inverse Fourier Transform"""
    with torch.no_grad():
        # tensor should have shape: (nc_out, nc_in, kx, ky)=(2*N, 2, N, kernel_size)
        nc_out, nc_in, N, kernel_size = tensor.shape

        for k in range(N):
            for n in range(N):
                tensor.data[k, 0, n, kernel_size // 2] = np.cos(2 * np.pi * n * k / N)
                tensor.data[k, 1, n, kernel_size // 2] = -np.sin(2 * np.pi * n * k / N)
                tensor.data[k + N, 0, n, kernel_size // 2] = np.sin(2 * np.pi * n * k / N)
                tensor.data[k + N, 1, n, kernel_size // 2] = np.cos(2 * np.pi * n * k / N)

        if norm == 'ortho':
            tensor.data[...] = tensor.data[...] / np.sqrt(N)

        return tensor

def get_refinement_block(model='automap_scae', in_channel=1, out_channel=1):
    if model == 'automap_scae':
        return nn.Sequential(nn.Conv2d(in_channel, 64, 5, 1, 2), nn.ReLU(True),
                             nn.Conv2d(64, 64, 5, 1, 2), nn.ReLU(True),
                             nn.ConvTranspose2d(64, out_channel, 7, 1, 3))
    elif model == 'simple':
        return nn.Sequential(nn.Conv2d(in_channel, 64, 3, 1, 1), nn.ReLU(True),
                             nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True),
                             nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True),
                             nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True),
                             nn.Conv2d(64, out_channel, 3, 1, 1))
    else:
        raise NotImplementedError


class GeneralisedIFT2Layer(nn.Module):

    def __init__(self, nrow, ncol,
                 nch_in, nch_int=None, nch_out=None,
                 kernel_size=1, nl=None,
                 init_fourier=True, init=None, bias=False, batch_norm=False,
                 share_tfxs=False, learnable=True):
        """Generalised domain transform layer

        The layer can be initialised as Fourier transform if nch_in == nch_int
        == nch_out == 2 and if init_fourier == True.

        It can also be initialised
        as Fourier transform plus noise by setting init_fourier == True and
        init == 'kaiming', for example.

        If nonlinearity nl is used, it is recommended to set bias = True

        One can use this layer as 2D Fourier transform by setting nch_in == nch_int
        == nch_out == 2 and learnable == False


        Parameters
        ----------
        nrow: int - the number of columns of input

        ncol: int - the number of rows of input

        nch_in: int - the number of input channels. One can put real & complex
        here, or put temporal coil channels, temporal frames, multiple
        z-slices, etc..

        nch_int: int - the number of intermediate channel after the transformation
        has been applied for each row. By default, this is the same as the input channel

        nch_out: int - the number of output channels. By default, this is the same as the input channel

        kernel_size: int - kernel size for second axis of 1d transforms

        init_fourier: bool - initialise generalised kernel with inverse fourier transform

        init_noise: str - initialise generalised kernel with standard initialisation. Option: ['kaiming', 'normal']

        nl: ('tanh', 'sigmoid', 'relu', 'lrelu') - add nonlinearity between two transformations. Currently only supports tanh

        bias: bool - add bias for each kernels

        share_tfxs: bool - whether to share two transformations

        learnable: bool

        """
        super(GeneralisedIFT2Layer, self).__init__()
        self.nrow = nrow
        self.ncol = ncol
        self.nch_in = nch_in
        self.nch_int = nch_int
        self.nch_out = nch_out
        self.kernel_size = kernel_size
        self.init_fourier = init_fourier
        self.init = init
        self.nl = nl

        if not self.nch_int:
            self.nch_int = self.nch_in

        if not self.nch_out:
            self.nch_out = self.nch_in

        # Initialise 1D kernels
        idft1 = torch.nn.Conv2d(self.nch_in, self.nch_int * self.nrow, (self.nrow, kernel_size),
                                padding=(0, kernel_size // 2), bias=bias)
        idft2 = torch.nn.Conv2d(self.nch_int, self.nch_out * self.ncol, (self.ncol, kernel_size),
                                padding=(0, kernel_size // 2), bias=bias)

        # initialise kernels
        init_noise_(idft1.weight, self.init)
        init_noise_(idft2.weight, self.init)

        if self.init_fourier:
            if not (self.nch_in == self.nch_int == self.nch_out == 2):
                raise ValueError

            if self.init:
                # scale the random weights to make it compatible with FFT basis
                idft1.weight.data = F.normalize(idft1.weight.data, dim=2)
                idft2.weight.data = F.normalize(idft2.weight.data, dim=2)

            init_fourier_(idft1.weight)
            init_fourier_(idft2.weight)

        self.idft1 = idft1
        self.idft2 = idft2

        # Allow sharing weights between two transforms if the input size are the same.
        if share_tfxs and nrow == ncol:
            self.idft2 = self.idft1

        self.learnable = learnable
        self.set_learnable(self.learnable)

        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn1 = torch.nn.BatchNorm2d(self.nch_int)
            self.bn2 = torch.nn.BatchNorm2d(self.nch_out)

    def forward(self, X):
        # input shape should be (batch_size, nc, nx, ny)
        #print("x size in dAUTOMAP GeneralisedIFT2Layer: ",X.size())
        batch_size = len(X)
        # first transform
        x_t = self.idft1(X)
        #print("x_t size in dAUTOMAP GeneralisedIFT2Layer: ",x_t.size())

        # reshape & transform
        x_t = x_t.reshape([batch_size, self.nch_int, self.nrow, self.ncol]).permute(0, 1, 3, 2)

        if self.batch_norm:
            x_t = self.bn1(x_t.contiguous())

        if self.nl:
            if self.nl == 'tanh':
                x_t = F.tanh(x_t)
            elif self.nl == 'relu':
                x_t = F.relu(x_t)
            elif self.nl == 'sigmoid':
                x_t = F.sigmoid(x_t)
            else:
                raise ValueError

        # second transform
        x_t = self.idft2(x_t)
        x_t = x_t.reshape([batch_size, self.nch_out, self.ncol, self.nrow]).permute(0, 1, 3, 2)

        if self.batch_norm:
            x_t = self.bn2(x_t.contiguous())


        return x_t

    def set_learnable(self, flag=True):
        self.learnable = flag
        self.idft1.weight.requires_grad = flag
        self.idft2.weight.requires_grad = flag


class dAUTOMAP(nn.Module):
    """
    Pytorch implementation of dAUTOMAP

    Decomposes the automap kernel into 2 Generalised "1D" transforms to make it scalable.
    """
    def __init__(self, input_shape, output_shape, tfx_params, tfx_params2=None):
        super(dAUTOMAP, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

        if tfx_params2 is None:
            tfx_params2 = tfx_params

        self.domain_transform = GeneralisedIFT2Layer(**tfx_params)
        self.domain_transform2 = GeneralisedIFT2Layer(**tfx_params2)
        self.refinement_block = get_refinement_block('automap_scae', input_shape[0], output_shape[0])

    def forward(self, x):
        """Assumes input to be (batch_size, 2, nrow, ncol)"""
        #print("x size in dAUTOMAP: ",x.size())
        x_mapped = self.domain_transform(x)
        x_mapped = F.tanh(x_mapped)
        #print("xmapped size in dAUTOMAP: ",x_mapped.size())
        x_mapped2 = self.domain_transform2(x_mapped)
        x_mapped2 = F.tanh(x_mapped2)
        #print("xmapped2 size in dAUTOMAP: ",x_mapped2.size())
        out = self.refinement_block(x_mapped2)
        #print("out size in dAUTOMAP: ",out.size())
        return out


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


def conv_block_cSE(n_ch, nd, nf=32, ks=3, dilation=1, bn=False, nl='lrelu', conv_dim=2, n_out=None):

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

    def conv_cSE():
        return cSELayer(nf)

    conv_1 = conv(n_ch, nf, ks, stride=1, padding=pad_conv, bias=True)
    conv_n = conv(nf, n_out, ks, stride=1, padding=pad_conv, bias=True)

    # relu
    nll = relu if nl == 'relu' else lrelu
    layers = [conv_1, conv_cSE(), nll()]
    for i in range(nd-2):
        if bn:
            layers.append(nn.BatchNorm2d(nf))
        layers += [conv_i(), conv_cSE(), nll()]

    layers += [conv_n]

    return nn.Sequential(*layers)

'''
x = torch.rand(1,2,150,150)
layer = conv_block(n_ch=2,nd=5,n_out=1)
y = layer(x)
print (y.shape)
'''
class ReconSynergyNetAblative(nn.Module):
    def __init__(self, args,**kwargs):
        super(ReconSynergyNetAblative, self).__init__()
        #dautomap_checkpoint_file = args.dautomap_model_path
        #da_checkpoint = torch.load(dautomap_checkpoint_file)
        #args = da_checkpoint['args']
        if args.dataset_type=='cardiac':
            patch_size = 150
        elif args.dataset_type=='mrbrain_t1':
            patch_size = 240
        elif args.dataset_type=='knee':
            patch_size = 320  
        else:
            patch_size = 256
        model_params = {
          'input_shape': (2, patch_size, patch_size),
          'output_shape': (1, patch_size, patch_size),
          'tfx_params': {
            'nrow': patch_size,
            'ncol': patch_size,
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
        self.dataset_type = args.dataset_type
        dautomap_model = dAUTOMAP(model_params['input_shape'],model_params['output_shape'],model_params['tfx_params'])#.to(args.device)
        
        unet_model = UnetModel(1,1,args.num_chans, args.num_pools, args.drop_prob)#.to(args.device)
        #unet_model = SCSEUnetModel(in_chans=1,out_chans=1,chans=32,num_pool_layers=4,drop_prob = 0,attention_type='cSE',reduction=16)            

        #srcnnlike_model = conv_block(n_ch=2,nd=5,n_out=1)#.to(args.device)
        srcnnlike_model = conv_block(n_ch=3,nd=5,n_out=1)#.to(args.device)
        #srcnnlike_model = conv_block_cSE(n_ch=2,nd=5,n_out=1)#.to(args.device)
        #srcnnlike_model = SCSEUnetModel(in_chans=2,out_chans=1,chans=32,num_pool_layers=4,drop_prob = 0,attention_type='cSE',reduction=16)            
        #srcnnlike_model = RecursiveDilatedBlock(n_ch=2,nd=5,n_out=1)#.to(args.device)
        #srcnnlike_model = HyperDenseNet()#.to(args.device)
        #srcnnlike_model = DenseNet()#.to(args.device)

        self.KI_layer = dautomap_model        
        self.II_layer = unet_model
        self.Re_layer = srcnnlike_model        
        

    def forward(self, x, xk):
        #xk = xk.permute(0,3,1,2)
        #print("x size in ReconSynergyNet: ",x.size())
        dautomap_pred = self.KI_layer(xk)
        #print(dautomap_pred.shape)
        if self.dataset_type=='cardiac':
            dautomap_pred = F.pad(dautomap_pred,(5,5,5,5),"constant",0)
        #print("dautomap_pred size in ReconSynergyNet: ",dautomap_pred.size())
        unet_pred     = self.II_layer(x)
        #print("unet_pred size in ReconSynergyNet: ",unet_pred.size())
        #pred_cat = torch.cat([unet_pred,dautomap_pred],dim=1)
        # converted to three channels as it is better to provide the undersampled image to refinement layer also.
        pred_cat = torch.cat([unet_pred,dautomap_pred,x],dim=1)
        recons = self.Re_layer(pred_cat)
        #recons = self.Re_layer(dautomap_pred,unet_pred)
        
        #print("recons size in ReconSynergyNet: ",recons.size())
        return recons


class ReconSynergyNet(nn.Module):
    def __init__(self, args,**kwargs):
        super(ReconSynergyNet, self).__init__()
        #dautomap_checkpoint_file = args.dautomap_model_path
        #da_checkpoint = torch.load(dautomap_checkpoint_file)
        #args = da_checkpoint['args']
        if args.dataset_type=='cardiac':
            patch_size = 150
        else:
            patch_size = 256
        model_params = {
          'input_shape': (2, patch_size, patch_size),
          'output_shape': (1, patch_size, patch_size),
          'tfx_params': {
            'nrow': patch_size,
            'ncol': patch_size,
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
        self.dataset_type = args.dataset_type
        dautomap_model = dAUTOMAP(model_params['input_shape'],model_params['output_shape'],model_params['tfx_params'])#.to(args.device)
        #if args.data_parallel:
        #    model = torch.nn.DataParallel(model)
        print ("Dautomap model initialized")
        #dautomap_model.load_state_dict(da_checkpoint['model'])

        print ("Dautomap model loaded")

        #unet_checkpoint_file = args.unet_model_path
        #unet_checkpoint = torch.load(unet_checkpoint_file)
        
        #unet_model = UnetModel(1,1,args.num_chans, args.num_pools, args.drop_prob)#.to(args.device)
        unet_model = SCSEUnetModel(in_chans=1,out_chans=1,chans=32,num_pool_layers=4,drop_prob = 0,attention_type='scSE',reduction=16)            
        #unet_model = UNetCSE(1,1)
        print ("UNet model initialized")
        #unet_model.load_state_dict(unet_checkpoint['model'])
        print ("Unet model loaded")

        #srcnnlike_checkpoint_file = args.srcnnlike_model_path
        #srcnnlike_checkpoint = torch.load(srcnnlike_checkpoint_file)
        #for k, v in srcnnlike_checkpoint.items():
        #    print("checkpoint items: ")
        #print(srcnnlike_checkpoint.keys())
        #srcnnlike_model = conv_block(n_ch=2,nd=5,n_out=1)#.to(args.device)
        srcnnlike_model = HyperDenseNet()#.to(args.device)
        #srcnnlike_model =  HyperDenseBlock2(n_ch=2)#.to(args.device)
        #srcnnlike_model = DenseBlock(n_ch=2, nd=5)#.to(args.device)
        #print ("srcnn model initialized")
        #srcnnlike_model.load_state_dict(srcnnlike_checkpoint['model'])
        print ("srcnn model loaded")
        #self.KI_layer = nn.ModuleList([dautomap_model])        
        #self.II_layer = nn.ModuleList([unet_model])
        #self.Re_layer = nn.ModuleList([srcnnlike_model])        

        self.KI_layer = dautomap_model        
        self.II_layer = unet_model
        self.Re_layer = srcnnlike_model        
        '''
        print (srcnnlike_model)
        print ("srcnn model initialized")
        new_state_dict = OrderedDict()
        for k, v in srcnnlike_checkpoint.items():
            name = k # remove 'module.' of dataparallel
            print("item name: ",type(v))
            print("item name: ",k)
            new_state_dict[name]=v
        '''
        #srcnnlike_model.load_state_dict(new_state_dict['model'])
        

    def forward(self, x, xk):
        #xk = xk.permute(0,3,1,2)
        #print("x size in ReconSynergyNet: ",x.size())
        dautomap_pred = self.KI_layer(xk)
        #print(dautomap_pred.shape)
        if self.dataset_type=='cardiac':
            dautomap_pred = F.pad(dautomap_pred,(5,5,5,5),"constant",0)
        #print("dautomap_pred size in ReconSynergyNet: ",dautomap_pred.size())
        unet_pred     = self.II_layer(x)
        #print("unet_pred size in ReconSynergyNet: ",unet_pred.size())
        #pred_cat = torch.cat([unet_pred,dautomap_pred],dim=1)
        #recons = self.Re_layer(pred_cat)
        recons = self.Re_layer(dautomap_pred,unet_pred)
        
        #print("recons size in ReconSynergyNet: ",recons.size())
        return recons


class DnCn(nn.Module):

    def __init__(self,args,n_channels=2, nc=3, nd=5, **kwargs):

        super(DnCn, self).__init__()

        self.nc = nc
        self.nd = nd
        self.dataset_type = args.dataset_type
        us_mask_path = os.path.join(args.usmask_path,'mask_{}.npy'.format(args.acceleration_factor))
        us_mask = torch.from_numpy(np.load(us_mask_path)).unsqueeze(2).unsqueeze(0).to(args.device)

        print('Creating D{}C{}'.format(nd, nc))
        conv_blocks = []
        dcs = []
        #checkpoint = torch.load(args.checkpoint)
        #conv_layer = conv_block


        for i in range(nc):
            #unetmodel = UnetModel(in_chans=1,out_chans=1,chans=args.num_chans,num_pool_layers=args.num_pools,drop_prob=args.drop_prob)
            #unetmodel.load_state_dict(checkpoint['model'])
            recon_synergynet_model = ReconSynergyNetAblative(args)
            print("recon synergy net: ",recon_synergynet_model.parameters)
            conv_blocks.append(recon_synergynet_model)
            dcs.append(DataConsistencyLayer(us_mask))

        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.dcs = dcs

    def forward(self,x,k):
        korig = k
        #print("korig size: ", korig.size())
        for i in range(self.nc):
            x_cnn = self.conv_blocks[i](x,k)
            #print("x size in DnCn: ", x.size())
            x = x + x_cnn # sriprabha : do we need this line ? 
            #print("x_cnn size in DnCn: ", x_cnn.size())
            #print("x size after addition  in DnCn: ", x.size())
            # x = self.dcs[i].perform(x, k, m)i
            xcrop=x
            if self.dataset_type=='cardiac':
                xcrop = x[:,:,5:x.shape[2]-5,5:x.shape[3]-5]
            #print("xcrop size: ", xcrop.size())
            x,k = self.dcs[i](xcrop,korig) # send original us_kspace for DC
            #print("xcorrect size: ", x.size(), " k corrected size: ", k.size())
            if self.dataset_type=='cardiac':
                x = F.pad(x,(5,5,5,5),"constant",0)
            k = k.permute(0,3,1,2)
        return x

