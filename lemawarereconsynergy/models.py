import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import variable, grad
import numpy as np
import os 
from unetmodel import UnetModel

########################### UNet for gradient estimation ################################



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
        lf=[]
        output = input
        H,W = input.shape[2],input.shape[3]
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            intout,output = layer(output)
            intoutup = F.upsample(intout,size=(H,W),mode='bilinear')
            outputup = F.upsample(output,size=(H,W),mode='bilinear')
            #print(intoutup.shape,outputup.shape)
            feat=torch.cat([intoutup,outputup],dim=1)
            lf.append(feat)
            stack.append(output)
            output = F.max_pool2d(output, kernel_size=2)

        _,output = self.conv(output)
        #print(output.shape)
        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            output = torch.cat([output, stack.pop()], dim=1)
            intout,output = layer(output)
            intoutup = F.upsample(intout,size=(H,W),mode='bilinear')
            outputup = F.upsample(output,size=(H,W),mode='bilinear')
            feat=torch.cat([intoutup,outputup],dim=1)
            lf.append(feat)
        finalfeat=torch.cat(lf,dim=1)    
        #print(thinfeat.shape)
     
        return finalfeat,self.conv2(output)





###################################################################################################

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




## Unet in Recon-synergy implementation 

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


class FastMRIUnetModel(nn.Module):
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

#####################################################################


################## Dautomap model ####################

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


###################################################################

########### Conv blocks #######################

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

#######################################


####################Convfeature block -- does convolution with the feature maps obtained from gradient maps######


class ConvFeatureBlock(nn.Module):
    def __init__(self,n_ch,nf=32):
        super(ConvFeatureBlock,self).__init__()

        self.conv1=nn.Sequential(nn.Conv2d(n_ch,nf,kernel_size=3,padding=1),nn.ReLU())
        self.conv2=nn.Sequential(nn.Conv2d(nf,nf,kernel_size=3,padding=1),nn.ReLU())
        self.conv3=nn.Sequential(nn.Conv2d(nf,nf,kernel_size=3,padding=1),nn.ReLU())
        self.conv4=nn.Sequential(nn.Conv2d(nf,nf,kernel_size=3,padding=1),nn.ReLU())
        self.conv5=nn.Conv2d(nf,1,kernel_size=3,padding=1)

        self.conv1x1_1=nn.Sequential(nn.Conv2d(nf+nf,nf,kernel_size=1),nn.ReLU())
        self.conv1x1_2=nn.Sequential(nn.Conv2d(nf+nf,nf,kernel_size=1),nn.ReLU())
        self.conv1x1_3=nn.Sequential(nn.Conv2d(nf+nf,nf,kernel_size=1),nn.ReLU())
        self.conv1x1_4=nn.Sequential(nn.Conv2d(nf+nf,nf,kernel_size=1),nn.ReLU())
      
    def forward(self, x, feat):

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

###################################################################################################


### ReconSynergyClass ###

class ReconSynergyNetAblative(nn.Module):
    def __init__(self, args,**kwargs):
        super(ReconSynergyNetAblative, self).__init__()

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

        dautomap_model = dAUTOMAP(model_params['input_shape'],model_params['output_shape'],model_params['tfx_params'])
        unet_model = FastMRIUnetModel(1,1,args.num_chans, args.num_pools, args.drop_prob)
        srcnnlike_model = conv_block(n_ch=3,nd=5,n_out=1)

        # load pretrained weights which are obtained from separately trainining the individual blocks to get the best possible result

        #unet_checkpoint     = torch.load(args.unet_model_path)
        #dautomap_checkpoint = torch.load(args.dautomap_model_path)
        #srcnnlike_checkpoint= torch.load(args.srcnnlike_model_path)

        #unet_model.load_state_dict(unet_checkpoint['model'])
        #dautomap_model.load_state_dict(dautomap_checkpoint['model'])
        #srcnnlike_model.load_state_dict(srcnnlike_checkpoint['model'])


        self.KI_layer = dautomap_model        
        self.II_layer = unet_model
        self.Re_layer = srcnnlike_model        
        

    def forward(self, x, xk):
        dautomap_pred = self.KI_layer(xk)

        if self.dataset_type=='cardiac':
            x = F.pad(x,(5,5,5,5),"constant",0)

        #if self.dataset_type=='cardiac':
        #    dautomap_pred = F.pad(dautomap_pred,(5,5,5,5),"constant",0)
        unet_pred     = self.II_layer(x)

        if self.dataset_type=='cardiac':
            unet_pred = unet_pred[:,:,5:155,5:155]
            x = x[:,:,5:155,5:155]

        # converted to three channels as it is better to provide the undersampled image to refinement layer also.
        pred_cat = torch.cat([unet_pred,dautomap_pred,x],dim=1)
        recons = self.Re_layer(pred_cat)
        
        return recons

###################


### ReconSynergyClass with gradient assistance  ###

class ReconSynergyNetAblativeFeature(nn.Module):
    def __init__(self, args,**kwargs):
        super(ReconSynergyNetAblativeFeature, self).__init__()

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

        dautomap_model = dAUTOMAP(model_params['input_shape'],model_params['output_shape'],model_params['tfx_params'])
        unet_model = FastMRIUnetModel(1,1,args.num_chans, args.num_pools, args.drop_prob)
        #srcnnlike_model = conv_block(n_ch=3,nd=5,n_out=1)
        srcnnlike_model =  ConvFeatureBlock(3)

        # load pretrained weights which are obtained from separately trainining the individual blocks to get the best possible result

        #unet_checkpoint     = torch.load(args.unet_model_path)
        #dautomap_checkpoint = torch.load(args.dautomap_model_path)
        #srcnnlike_checkpoint= torch.load(args.srcnnlike_model_path)

        #unet_model.load_state_dict(unet_checkpoint['model'])
        #dautomap_model.load_state_dict(dautomap_checkpoint['model'])
        #srcnnlike_model.load_state_dict(srcnnlike_checkpoint['model'])


        self.KI_layer = dautomap_model        
        self.II_layer = unet_model
        self.Re_layer = srcnnlike_model        


    def forward(self, x, xk,feat):

        dautomap_pred = self.KI_layer(xk)
  
        if self.dataset_type=='cardiac':
            x = F.pad(x,(5,5,5,5),"constant",0)

        unet_pred     = self.II_layer(x)

        if self.dataset_type=='cardiac':
            unet_pred = unet_pred[:,:,5:155,5:155]
            x = x[:,:,5:155,5:155]

        # converted to three channels as it is better to provide the undersampled image to refinement layer also.

        pred_cat = torch.cat([unet_pred,dautomap_pred,x],dim=1)
        recons = self.Re_layer(pred_cat,feat)
        
        return recons

###################





######## DC-RSN ###########

class DnCn(nn.Module):

    def __init__(self,args,n_channels=2, nc=5, nd=5, **kwargs):

        super(DnCn, self).__init__()

        self.nc = nc
        self.nd = nd
        self.dataset_type = args.dataset_type
        us_mask_path = os.path.join(args.usmask_path,'mask_{}.npy'.format(args.acceleration_factor))
        us_mask = torch.from_numpy(np.load(us_mask_path)).unsqueeze(2).unsqueeze(0).to(args.device)

        print('Creating D{}C{}'.format(nd, nc))
        conv_blocks = []
        dcs = []


        for i in range(nc):
            recon_synergynet_model = ReconSynergyNetAblative(args)
            conv_blocks.append(recon_synergynet_model)
            dcs.append(DataConsistencyLayer(us_mask))

        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.dcs = dcs

    def forward(self,x,k):
        korig = k

        for i in range(self.nc):

            x = self.conv_blocks[i](x,k)
            #xcrop=x

            #if self.dataset_type=='cardiac':
            #    xcrop = x[:,:,5:x.shape[2]-5,5:x.shape[3]-5]

            x,k = self.dcs[i](x,korig) 

            #if self.dataset_type=='cardiac':
            #    x = F.pad(x,(5,5,5,5),"constant",0)

            k = k.permute(0,3,1,2)

        return x

########################


################### DC-CNN which uses class to create conv layers, gradient feature is used as additional information #########################


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

        self.conv1x1=nn.Conv2d(1472,32,kernel_size=1)

        for i in range(nc):
            conv_feature_block= ReconSynergyNetAblativeFeature(args)
            conv_blocks.append(conv_feature_block)
            dcs.append(DataConsistencyLayer(us_mask))

        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.dcs = dcs

    def forward(self,x,k,feat):

        thinfeat=self.conv1x1(feat)
          
        korig = k 

        for i in range(self.nc):

            x_cnn = self.conv_blocks[i](x,k,thinfeat)
            #xcrop = x 

            #if self.dataset_type=='cardiac':
            #    xcrop = x[:,:,5:x.shape[2]-5,5:x.shape[3]-5]

            x,k = self.dcs[i](x,korig) 

            #if self.dataset_type=='cardiac':
            #    x = F.pad(x,(5,5,5,5),"constant",0)
            k = k.permute(0,3,1,2)

        return x


############################################

