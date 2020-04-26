import torch
from torch import nn
from torch.nn import functional as F
import os 
import numpy as np

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

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob)
        )


        self.layer2 = nn.Sequential(
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
        intout=self.layer1(input)
        out=self.layer2(intout)
        return intout,out

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
        layerfeatures=[]
        H,W=input.shape[2],input.shape[3]
        print(H,W)
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            #print("1",output.shape)
            intd,output = layer(output)
            print("up",intd.shape,output.shape)
            feat=torch.cat([intd,output],dim=1)
            featup=F.upsample(feat,size=(H,W),mode='bilinear')
            print("feat: ",featup.shape,feat.shape)
            layerfeatures.append(featup)
            stack.append(output)
            output = F.max_pool2d(output, kernel_size=2)
        intm,output = self.conv(output)

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            output = torch.cat([output, stack.pop()], dim=1)
            intu,output = layer(output)
            feat=torch.cat([intu,output],dim=1)
            featdown=F.upsample(feat,size=(H,W),mode='bilinear')
            layerfeatures.append(featdown)
            #print("down",intu.shape,output.shape)
        lemfeatures=torch.cat(layerfeatures, dim=1)
        print(lemfeatures.shape)
        return self.conv2(output),lemfeatures

    
class DataConsistencyLayer(nn.Module):

    def __init__(self,mask_path,acc_factor,device):
        
        super(DataConsistencyLayer,self).__init__()

        print (mask_path)
        mask_path = os.path.join(mask_path,'mask_{}.npy'.format(acc_factor))
        self.mask = torch.from_numpy(np.load(mask_path)).unsqueeze(2).unsqueeze(0).to(device)

    def forward(self,us_kspace,predicted_img):

        # us_kspace     = us_kspace[:,0,:,:]
        predicted_img = predicted_img[:,0,:,:]
        
        kspace_predicted_img = torch.rfft(predicted_img,2,True,False).double()
        # print (us_kspace.shape,predicted_img.shape,kspace_predicted_img.shape,self.mask.shape)
        
        updated_kspace1  = self.mask * us_kspace 
        updated_kspace2  = (1 - self.mask) * kspace_predicted_img

        

        updated_kspace   = updated_kspace1[:,0,:,:,:] + updated_kspace2
        
        
        updated_img    = torch.ifft(updated_kspace,2,True) 
        
        update_img_abs = torch.sqrt(updated_img[:,:,:,0]**2 + updated_img[:,:,:,1]**2)
        
        update_img_abs = update_img_abs.unsqueeze(1)
        
        return update_img_abs.float()

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
 
