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

        predicted_img = predicted_img.permute(0,2,3,1)
        kspace_predicted_img = torch.fft(predicted_img,2,True)
        us_kspace = us_kspace.permute(0,2,3,1)
        #print (us_kspace.shape,predicted_img.shape,kspace_predicted_img.shape,self.us_mask.shape)
        #print (us_kspace.dtype,predicted_img.dtype,kspace_predicted_img.dtype,self.us_mask.dtype)
        
        updated_kspace1  = self.us_mask * us_kspace 
        updated_kspace2  = (1 - self.us_mask) * kspace_predicted_img

        updated_kspace   = updated_kspace1 + updated_kspace2
        
        updated_img    = torch.ifft(updated_kspace,2,True) 

        updated_img = updated_img.permute(0,3,1,2)
        
        return updated_img 


class DnCn(nn.Module):

    def __init__(self,args,n_channels=2, nc=5, nd=5,**kwargs):

        super(DnCn, self).__init__()

        self.nc = nc
        self.nd = nd

        us_mask_path = os.path.join(args.usmask_path,'mask_{}.npy'.format(args.acceleration_factor))
        us_mask = torch.from_numpy(np.load(us_mask_path)).unsqueeze(2).unsqueeze(0).float().to(args.device)

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
