import torch
import torch.nn as nn
from torch.autograd import Variable, grad
import numpy as np
import os 
from torchvision import models
import functools


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


# def data_consistency(k, k0, mask, noise_lvl=None):
#     """
#     k    - input in k-space
#     k0   - initially sampled elements in k-space
#     mask - corresponding nonzero location
#     """
#     v = noise_lvl
#     if v:  # noisy case
#         out = (1 - mask) * k + mask * (k + v * k0) / (1 + v)
#     else:  # noiseless case
#         out = (1 - mask) * k + mask * k0
#     return out


# class DataConsistencyInKspace(nn.Module):
#     """ Create data consistency operator
#     Warning: note that FFT2 (by the default of torch.fft) is applied to the last 2 axes of the input.
#     This method detects if the input tensor is 4-dim (2D data) or 5-dim (3D data)
#     and applies FFT2 to the (nx, ny) axis.
#     """

#     def __init__(self, noise_lvl=None, norm='ortho'):
#         super(DataConsistencyInKspace, self).__init__()
#         self.normalized = norm == 'ortho'
#         self.noise_lvl = noise_lvl

#     def forward(self, *input, **kwargs):
#         return self.perform(*input)

#     def perform(self, x, k0, mask):
#         """
#         x    - input in image domain, of shape (n, 2, nx, ny[, nt])
#         k0   - initially sampled elements in k-space
#         mask - corresponding nonzero location
#         """

#         if x.dim() == 4: # input is 2D
#             x    = x.permute(0, 2, 3, 1)
#             k0   = k0.permute(0, 2, 3, 1)
#             mask = mask.permute(0, 2, 3, 1)
#         elif x.dim() == 5: # input is 3D
#             x    = x.permute(0, 4, 2, 3, 1)
#             k0   = k0.permute(0, 4, 2, 3, 1)
#             mask = mask.permute(0, 4, 2, 3, 1)

#         k = torch.fft(x, 2, normalized=self.normalized)
#         out = data_consistency(k, k0, mask, self.noise_lvl)
#         x_res = torch.ifft(out, 2, normalized=self.normalized)

#         if x.dim() == 4:
#             x_res = x_res.permute(0, 3, 1, 2)
        #         elif x.dim() == 5:
#             x_res = x_res.permute(0, 4, 2, 3, 1)

#         return x_res

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

class Discriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc=1, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(Discriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)
        #self.fc = nn.Linear(30*30,1)#kirby,(256,256)
        self.fc = nn.Linear(16*16,1) #cardiac,(160,160)
        #self.fc = nn.Linear(28*28,1) #mrbrain(240,240)

    def forward(self, input):
        """Standard forward."""
        out1 = self.model(input)
        #print (out1.shape)
        #out2 = self.fc(out1.view(-1,30*30))
        #out2 = self.fc(out1.view(-1,18*18)) #cardiac 
        
        out2 = self.fc(out1.view(-1,out1.shape[-2]*out1.shape[-1]))
        return out2
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



