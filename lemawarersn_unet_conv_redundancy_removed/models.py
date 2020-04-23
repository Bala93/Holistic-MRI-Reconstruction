import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import variable, grad
import numpy as np
import os 
from dautomap import dAUTOMAP, dAUTOMAPFeat
from fastmriunet import FastMRIUnetModel
from unetmodels import *
from utils import *
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
            #intoutup = F.upsample(intout,size=(H,W),mode='bilinear')
            outputup = F.upsample(output,size=(H,W),mode='bilinear')
            print(output.shape)
            #feat=torch.cat([intoutup,outputup],dim=1)
            lf.append(outputup)
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
        print(output.shape)
        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            output = torch.cat([output, stack.pop()], dim=1)
            intout,output = layer(output)
            #intoutup = F.upsample(intout,size=(H,W),mode='bilinear')
            outputup = F.upsample(output,size=(H,W),mode='bilinear')
            print(output.shape)
            #feat=torch.cat([intoutup,outputup],dim=1)
            lf.append(outputup)
        finalfeat=torch.cat(lf,dim=1)    
        #print(thinfeat.shape)
     
        return finalfeat,self.conv2(output)

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



###################################################################################################

## Dataconsistency implementation

class DataConsistencyLayer(nn.Module):

    #def __init__(self,us_mask):
    def __init__(self):
        
        super(DataConsistencyLayer,self).__init__()

        #self.us_mask = us_mask 
        #self.us_mask = self.us_mask.float() 

    def forward(self,predicted_img,us_kspace,us_mask):

        predicted_img = predicted_img[:,0,:,:]
        
        kspace_predicted_img = torch.rfft(predicted_img,2,True,False)
        us_kspace = us_kspace.permute(0,2,3,1)
        us_mask = us_mask.permute(0,2,3,1)
        #updated_kspace1  = self.us_mask * us_kspace 
        #print (us_kspace.shape,kspace_predicted_img.shape,us_mask.shape)
        updated_kspace1  = us_mask * us_kspace 
        #updated_kspace2  = (1 - self.us_mask) * kspace_predicted_img
        updated_kspace2  = (1 - us_mask) * kspace_predicted_img

        updated_kspace   = updated_kspace1 + updated_kspace2
        
        
        updated_img    = torch.ifft(updated_kspace,2,True) 
        
        update_img_abs = updated_img[:,:,:,0]
        
        update_img_abs = update_img_abs.unsqueeze(1)
        
        return update_img_abs.float(), updated_kspace

##################################################################

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

        #self.conv1x1=nn.Conv2d(1984,nf,kernel_size=1)
        #self.conv1x1=nn.Conv2d(992,nf,kernel_size=1)
        self.conv1=nn.Sequential(nn.Conv2d(n_ch,nf,kernel_size=3,padding=1),nn.ReLU())
        self.conv2=nn.Sequential(nn.Conv2d(nf+32,nf,kernel_size=3,padding=1),nn.ReLU())
        self.conv3=nn.Sequential(nn.Conv2d(nf+32,nf,kernel_size=3,padding=1),nn.ReLU())
        self.conv4=nn.Sequential(nn.Conv2d(nf+32,nf,kernel_size=3,padding=1),nn.ReLU())
        self.conv5=nn.Conv2d(nf+32,1,kernel_size=3,padding=1)

      
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
        unet_model = FastMRIUnetModel(1,1,32,4,0)
        srcnnlike_model = conv_block(n_ch=3,nd=5,n_out=1)

        # load pretrained weights which are obtained from separately trainining the individual blocks to get the best possible result

#        unet_checkpoint     = torch.load(args.unet_model_path)
#        dautomap_checkpoint = torch.load(args.dautomap_model_path)
#        srcnnlike_checkpoint= torch.load(args.srcnnlike_model_path)
#
#        unet_model.load_state_dict(unet_checkpoint['model'])
#        dautomap_model.load_state_dict(dautomap_checkpoint['model'])
#        srcnnlike_model.load_state_dict(srcnnlike_checkpoint['model'])


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

        #dautomap_model = dAUTOMAP(model_params['input_shape'],model_params['output_shape'],model_params['tfx_params'])
        dautomap_model = dAUTOMAPFeat(model_params['input_shape'],model_params['output_shape'],model_params['tfx_params'])
        #unet_model = FastMRIUnetModel(1,1,args.num_chans, args.num_pools, args.drop_prob)
        #unet_model = UnetModelAssistEverywhere(1,1,args.num_chans, args.num_pools, args.drop_prob)
        unet_model = UnetModelAssistLatentDecoder(1,1,args.num_chans, args.num_pools, args.drop_prob)
        #srcnnlike_model = conv_block(n_ch=3,nd=5,n_out=1)
        srcnnlike_model =  ConvFeatureBlock(3)

        # load pretrained weights which are obtained from separately trainining the individual blocks to get the best possible result
        #unet_checkpoint     = torch.load(args.unet_model_path)
        #dautomap_checkpoint = torch.load(args.dautomap_model_path)
        #srcnnlike_checkpoint= torch.load(args.srcnnlike_model_path)
        #print (unet_checkpoint['model'].keys(),dautomap_checkpoint['model'].keys())

        #upd_unet_checkpoint = fix_module_from_dataparallel(unet_checkpoint['model'])
        #print (upd_unet_checkpoint.keys(),dautomap_checkpoint['model'].keys())
        #unet_model.load_state_dict(upd_unet_checkpoint)
        #dautomap_model.load_state_dict(dautomap_checkpoint['model'])
        #srcnnlike_model.load_state_dict(srcnnlike_checkpoint['model_re'])

        #for param in dautomap_model.parameters():
        #    param.requires_grad = False 

        #for param in unet_model.parameters():
        #    param.requires_grad = False 


        self.KI_layer = dautomap_model        
        self.II_layer = unet_model
        self.Re_layer = srcnnlike_model        


    def forward(self, x, xk,feat):

        #print (x.shape,xk.shape,feat.shape)
        dautomap_pred = self.KI_layer(xk,feat)
  
        if self.dataset_type=='cardiac':
            x = F.pad(x,(5,5,5,5),"constant",0)
            feat = F.pad(feat,(5,5,5,5),"constant",0)

        #print (x.shape,xk.shape,feat.shape)
        unet_pred     = self.II_layer(x,feat)

        if self.dataset_type=='cardiac':
            unet_pred = unet_pred[:,:,5:155,5:155]
            x = x[:,:,5:155,5:155]
            feat = feat[:,:,5:155,5:155]
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

    def __init__(self,args,n_channels=2, nc=1, nd=5,**kwargs):

        super(DnCnFeature, self).__init__()

        self.nc = nc
        self.nd = nd

        ''' commented to check data parallel by using it outside of class 
        #us_mask_path = os.path.join(args.usmask_path,'mask_{}.npy'.format(args.acceleration_factor))
        #us_mask = torch.from_numpy(np.load(us_mask_path)).unsqueeze(2).unsqueeze(0).to(args.device)
        #print (us_mask.device)
        '''

        print('Creating D{}C{}'.format(nd, nc))
        conv_blocks = []
        dcs = []

        #self.conv1x1=nn.Conv2d(1472,32,kernel_size=1)
        #conv1x1_checkpoint = torch.load(args.srcnnlike_model_path)
        #self.conv1x1.load_state_dict(conv1x1_checkpoint['model_conv1x1']) 

        for i in range(nc):
            conv_feature_block= ReconSynergyNetAblativeFeature(args)
            conv_blocks.append(conv_feature_block)
            #dcs.append(DataConsistencyLayer(us_mask))
            dcs.append(DataConsistencyLayer())

        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.dcs = dcs

    def forward(self,x,k,thinfeat,us_mask):

        #thinfeat=self.conv1x1(feat)
          
        korig = k 

        for i in range(self.nc):

            #print ("\t In model Befor x1:",x.shape,k.shape)
            x= self.conv_blocks[i](x,k,thinfeat)
            #xcrop = x 

            #if self.dataset_type=='cardiac':
            #    xcrop = x[:,:,5:x.shape[2]-5,5:x.shape[3]-5]

            #print ("\t In model After x1:",x.shape,k.shape)

            #x,k = self.dcs[i](x,korig,us_mask) 


            #print ("\t In model x2:",x.shape,k.shape)
            #if self.dataset_type=='cardiac':
            #    x = F.pad(x,(5,5,5,5),"constant",0)
            #k = k.permute(0,3,1,2)

        return x


############################################

class DnCnFeatureLoop(nn.Module):

    def __init__(self,args,n_channels=2, nc=5, nd=5,**kwargs):

        super(DnCnFeatureLoop, self).__init__()
        nc_value = {'cardiac':5,'kirby90':3}
        self.nc = nc_value[args.dataset_type]
        self.nd = nd

        print('Creating D{}C{}loop'.format(self.nd, self.nc))
        conv_blocks = []
        dcs = []
        #lemawarersnnc1_checkpoint = torch.load(args.lemaware_rsn_nc1_path)
        self.conv1x1=nn.Conv2d(512,32,kernel_size=1)
        #print ("############: ",lemawarersnnc1_checkpoint['model'].keys())
        for i in range(nc):
            lemawarersnnc1_block = DnCnFeature(args, n_channels,1,5,**kwargs)
            #lemawarersnnc1_block.load_state_dict(fix_module_from_dataparallel(lemawarersnnc1_checkpoint['model']))
            #print (lemawarersnnc1_checkpoint['model'])
            conv_blocks.append(lemawarersnnc1_block)
            dcs.append(DataConsistencyLayer())

        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.dcs = dcs

    def forward(self,x,k,feat,us_mask):
        korig = k 
        thinfeat = self.conv1x1(feat)
        #print (thinfeat.shape)
        #thinfeat = F.upsample(thinfeat,size=(256,256),mode='bilinear')
        
        #print("thinfeat device: ",thinfeat.shape)

        for i in range(self.nc):
            #print("thinfeat device: ",thinfeat.device)
            x = self.conv_blocks[i](x,k,thinfeat,us_mask)
            x,k = self.dcs[i](x,korig,us_mask) 
            k = k.permute(0,3,1,2)
        return x


