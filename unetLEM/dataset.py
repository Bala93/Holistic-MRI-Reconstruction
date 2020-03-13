import pathlib
import random
import numpy as np
import h5py
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from skimage import feature
import os 
from utils import npComplexToTorch
from models import LEM
class SliceData(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    #def __init__(self, root, acc_factor,dataset_type,mask_path): # acc_factor can be passed here and saved as self variable
    def __init__(self, root, acc_factor,dataset_type): # acc_factor can be passed here and saved as self variable
        # List the h5 files in root 
        files = list(pathlib.Path(root).iterdir())
        self.examples = []
        self.acc_factor = acc_factor 
        self.dataset_type = dataset_type
        self.key_img = 'img_volus_{}'.format(self.acc_factor)
        self.key_kspace = 'kspace_volus_{}'.format(self.acc_factor)
        self.lemfunc = LEM()#  device
        #mask_path = os.path.join(mask_path,'mask_{}.npy'.format(acc_factor))
        #self.mask = np.load(mask_path)
        #print (files)

        for fname in sorted(files):
            with h5py.File(fname,'r') as hf:
                #print (hf.keys())
                fsvol = hf['volfs']
                num_slices = fsvol.shape[2]
                self.examples += [(fname, slice) for slice in range(num_slices)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # Index the fname and slice using the list created in __init__
        
        fname, slice = self.examples[i] 
        # Print statements 
        #print (fname,slice)
    
        with h5py.File(fname, 'r') as data:

            #input_img  = data[self.key_img][:,:,slice]
            #input_kspace  = data[self.key_kspace][:,:,slice]
            #input_kspace = npComplexToTorch(input_kspace)
    
            target = data['volfs'][:,:,slice].astype(np.float64)


            target_tensor = torch.from_numpy(target).unsqueeze(0).unsqueeze(0)
            targetlem = self.lemfunc(target_tensor)
            #print (target_tensor.shape, targetlem.shape)
            target_tensor = target_tensor.squeeze(0).squeeze(0)
            targetlem = targetlem.squeeze(0).squeeze(0)

            if self.dataset_type == 'cardiac':
                # Cardiac dataset should be padded,150 becomes 160. # this can be commented for kirby brain 
                target_tensor = F.pad(target_tensor,(5,5,5,5),"constant",0)
                targetlem = F.pad(targetlem,(5,5,5,5),"constant",0)


            #print (input_img.shape,input_kspace.shape,target.shape)

            #kspace_cmplx = np.fft.fft2(target,norm='ortho')
            #uskspace_cmplx = kspace_cmplx * self.mask
            #zf_img = np.abs(np.fft.ifft2(uskspace_cmplx,norm='ortho'))
            
            # Print statements
            #print (input.shape,target.shape)
            #return torch.from_numpy(zf_img), torch.from_numpy(target)
            # print (input_img.dtype,input_kspace.dtype,target.dtype)
            #print (torch.from_numpy(input_img), input_kspace, torch.from_numpy(target))
            return target_tensor,targetlem
            
class SliceDataDev(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    #def __init__(self, root,acc_factor,dataset_type,mask_path):
    def __init__(self, root,acc_factor,dataset_type):

        # List the h5 files in root 
        files = list(pathlib.Path(root).iterdir())
        self.examples = []
        self.acc_factor = acc_factor
        self.dataset_type = dataset_type
        print("self.acc_factor: ", self.acc_factor)
        self.key_img = 'img_volus_{}'.format(self.acc_factor)
        self.key_kspace = 'kspace_volus_{}'.format(self.acc_factor)
        self.lemfunc = LEM()#  device

        #mask_path = os.path.join(mask_path,'mask_{}.npy'.format(acc_factor))
        #self.mask = np.load(mask_path)

        for fname in sorted(files):
            with h5py.File(fname,'r') as hf:
                #print(hf.keys())
                fsvol = hf['volfs']
                num_slices = fsvol.shape[2]
                self.examples += [(fname, slice) for slice in range(num_slices)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # Index the fname and slice using the list created in __init__
        
        fname, slice = self.examples[i]
        # Print statements 
        #print (type(fname),slice)
    
        with h5py.File(fname, 'r') as data:

            #input_img  = data[self.key_img][:,:,slice]
            #input_kspace  = data[self.key_kspace][:,:,slice]
            #input_kspace = npComplexToTorch(input_kspace)
            target = data['volfs'][:,:,slice]
            target_tensor = torch.from_numpy(target).unsqueeze(0).unsqueeze(0)
            targetlem = self.lemfunc(target_tensor)
            #print (target_tensor.shape, targetlem.shape)
            target_tensor = target_tensor.squeeze(0).squeeze(0)
            targetlem = targetlem.squeeze(0).squeeze(0)
 
            #kspace_cmplx = np.fft.fft2(target,norm='ortho')
            #uskspace_cmplx = kspace_cmplx * self.mask
            #zf_img = np.abs(np.fft.ifft2(uskspace_cmplx,norm='ortho'))
 

            #if self.dataset_type == 'cardiac':
                # Cardiac dataset should be padded,150 becomes 160. # this can be commented for kirby brain 
                #input_img  = np.pad(input_img,(5,5),'constant',constant_values=(0,0))
                #target = np.pad(target,(5,5),'constant',constant_values=(0,0))

            # Print statements
            #print (input.shape,target.shape)
            return target_tensor, targetlem,str(fname.name),slice
            #return torch.from_numpy(zf_img), torch.from_numpy(target),str(fname.name),slice
