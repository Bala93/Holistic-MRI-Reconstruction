import torch
import torch.nn as nn
from torch.autograd import Variable, grad
import numpy as np
import os 


def lrelu():
    return nn.LeakyReLU(0.01, inplace=True)

def relu():
    return nn.ReLU(inplace=True)

class DenseConnectionLayerSingle(nn.Module):
    def __init__(self, in_chans):
        super().__init__()
        self.layers= nn.Conv2d(in_chans, 1, kernel_size=3, padding=1)

    def forward(self,x):
        #print(x.shape)
        xout = self.layers(x) 
        return xout

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
        conv_n = conv(nf+nd-1, n_out, ks, stride=1, padding=pad_conv, bias=True)
        # relu
        nll = relu if nl == 'relu' else lrelu
        self.denseConnlayers = nn.ModuleList([])
        self.layers = nn.ModuleList([nn.Sequential(conv_1, nll())])
        #self.layers = nn.ModuleList([conv_1, nll()])

        for i in range(1,nd-1):
            self.denseConnlayers += [DenseConnectionLayerSingle(nf)]
            self.layers += [nn.Sequential(conv_i(nf+i), nll())]

        self.denseConnlayers += [DenseConnectionLayerSingle(nf)]
        self.layers += [conv_n]

    def forward(self,x,nd):
        flist=[]
        xtemp = self.layers[0](x) # first conv that takes image input, gives out 32 feature maps
        feat = xtemp
        catlist=[]
        for i in range(1,nd-1):
            x1 = self.denseConnlayers[i-1](feat) #takes i+1 list each having 32 feature maps gives i+1 feature maps as output 
            catlist.append(x1)
            xfeat = torch.cat(catlist,dim=1)
            x3 = torch.cat([xtemp,xfeat],dim=1)
            x4 = self.layers[i](x3) # takes ith conv layer 32+i feature maps
            feat=x4
            xtemp=x4
        #print("dense len: ", len(self.denseConnlayers))
        x1 = self.denseConnlayers[nd-2](feat) #takes i+1 list each having 32 feature maps gives i+1 feature maps as output 
        catlist.append(x1)
        xfeat = torch.cat(catlist,dim=1)
        x3 = torch.cat([xtemp,xfeat],dim=1) 
        #print("convlen: ", len(self.layers)) 
        out = self.layers[nd-1](x3) # takes ith conv layer 32+i feature maps
        return out
'''
    def __init__(self, n_ch, nd=5, nf=32, ks=3, dilation=1, bn=False, nl='lrelu', conv_dim=2, n_out=1):
        super().__init__()

        # convolution dimension (2D or 3D)
        if conv_dim == 2:
            conv = nn.Conv2d
        else:
            conv = nn.Conv3d

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


    def forward(self,x,nd):
        #print("x entering: ", x.shape) #[1, 1, 150, 150]
        flist=[]
        xtemp = self.layers[0](x) # first conv that takes image input, gives out 32 feature maps
        xtemp = self.layers[1](xtemp) # 1st ReLU
        #print("x temp after first conv+ReLU: ", xtemp.shape)#[1, 32, 150, 150])
        x = x.unsqueeze(0)
        flist.append(xtemp)
        for i in range(1,nd):
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
        x1 = self.denseConnlayers[nd-1](featureList,nd) #takes i+1 list each having 32 feature maps gives i+1 feature maps as output 
        #print("final x1 output of denseconn shape: ", x1.shape, "nd value: ",nd) #[1, 1, 1, 150, 150]
        xtemp = xtemp.unsqueeze(0)
        #print("final xtemp shape after unsqueeze: ", xtemp.shape) #[1, 1, 1, 150, 150]
        x1 = x1.permute(2,1,0,3,4)
        #print("final x temp shape: ", xtemp.shape, "x1.shape: ", x1.shape)
        x3 = torch.cat([xtemp,x1],dim=2) 
        x3 = x3.squeeze(0)
        #print("final x3 shape: ", x3.shape)
        #print("final self.layers[2*1]",self.layers[2*nd], " 2*i: ",2*nd)
        out = self.layers[2*nd](x3) # takes ith conv layer 32+i feature maps
        return out


'''
class DataConsistencyLayer(nn.Module):

    def __init__(self,us_mask):
        
        super(DataConsistencyLayer,self).__init__()

        self.us_mask = us_mask 

    def forward(self,predicted_img,us_kspace):

        # us_kspace     = us_kspace[:,0,:,:]
        predicted_img = predicted_img[:,0,:,:]
        
        kspace_predicted_img = torch.rfft(predicted_img,2,True,False).double()
        #print (us_kspace.shape,predicted_img.shape,kspace_predicted_img.shape,self.us_mask.shape)
        
        #print("us_kspace: ",us_kspace)
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
            conv_blocks.append(DenseBlock(n_ch=n_channels))
            dcs.append(DataConsistencyLayer(us_mask))

        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.dcs = dcs
        self.outputlayer = nn.Conv2d(nc, 1, kernel_size=3, stride=1, padding=1, bias=True)
    def forward(self,x,k):
        outputs=[]
        for i in range(self.nc):
            x_cnn = self.conv_blocks[i](x,self.nd)
            x = x + x_cnn
            # x = self.dcs[i].perform(x, k, m)
            x = self.dcs[i](x,k)
            #print("x shape: ",x.shape)
            outputs.append(x)
        #return x
        stackedout = torch.cat(outputs,dim=1)
        #print("stackedoutput shape: ", stackedout.shape)
        out = self.outputlayer(stackedout)
        return out

'''
k=4
x = torch.rand(1,1,150,150)
layer = DenseBlock(n_ch=1)
print (layer)
y = layer(x,k)
x = torch.rand(k,1,32,150,150)
#layer = DenseConnectionLayer(32,k)
#y = layer(x,k)
print (y.shape)

'''


