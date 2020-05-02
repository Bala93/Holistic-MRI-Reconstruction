import pathlib
import random
import numpy as np
import h5py
from torch.utils.data import Dataset
import torch
import os 


class SliceData(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root, acc_factor,dataset_type): # acc_factor can be passed here and saved as self variable
        # List the h5 files in root 
        files = list(pathlib.Path(root).iterdir())
        self.examples = []
        self.acc_factor = acc_factor 
        self.dataset_type = dataset_type
        self.key_img = 'img_volus_{}'.format(self.acc_factor)
        self.key_kspace = 'kspace_volus_{}'.format(self.acc_factor)
        mask_path = os.path.join('/data/balamurali/Holistic-MRI-Reconstruction/us_masks/calgary/cartesian','mask_{}.npy'.format(acc_factor))
        mask = np.expand_dims(np.expand_dims(np.load(mask_path),axis=0),axis=-1)
        self.mask = np.repeat(mask,2,axis=-1)
        self.sensitivity = np.stack([np.ones([1,256,256]),np.zeros([1,256,256])],axis=-1)

        for fname in sorted(files):
            with h5py.File(fname,'r') as hf:
                fsvol = hf['volfs']
                num_slices = fsvol.shape[0]
                self.examples += [(fname, slice) for slice in range(num_slices)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # Index the fname and slice using the list created in __init__
        
        fname, slice = self.examples[i] 
        # Print statements 
        #print (fname,slice)
    
        with h5py.File(fname, 'r') as data:

            input_img  = data[self.key_img][slice,:,:,:]
            input_kspace  = data[self.key_kspace][slice,:,:,:]
            target = data['volfs'][slice,:,:,:]

            #input_img = np.transpose(input_img,[2,0,1])
            #input_kspace = np.transpose(input_kspace,[2,0,1])
            #target = np.transpose(target,[2,0,1])

            return torch.from_numpy(target),torch.from_numpy(input_img), torch.from_numpy(input_kspace), torch.from_numpy(input_kspace),torch.from_numpy(self.mask),torch.from_numpy(self.sensitivity)


 
class SliceDataDev(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root, acc_factor,dataset_type): # acc_factor can be passed here and saved as self variable
        # List the h5 files in root 
        files = list(pathlib.Path(root).iterdir())
        self.examples = []
        self.acc_factor = acc_factor 
        self.dataset_type = dataset_type
        self.key_img = 'img_volus_{}'.format(self.acc_factor)
        self.key_kspace = 'kspace_volus_{}'.format(self.acc_factor)
        mask_path = os.path.join('/data/balamurali/Holistic-MRI-Reconstruction/us_masks/calgary/cartesian','mask_{}.npy'.format(self.acc_factor))
        mask = np.expand_dims(np.expand_dims(np.load(mask_path),axis=0),axis=-1)
        self.mask = np.repeat(mask,2,axis=-1)
        self.sensitivity = np.stack([np.ones([1,256,256]),np.zeros([1,256,256])],axis=-1)

        for fname in sorted(files):
            with h5py.File(fname,'r') as hf:
                fsvol = hf['volfs']
                num_slices = fsvol.shape[0]
                self.examples += [(fname, slice) for slice in range(num_slices)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # Index the fname and slice using the list created in __init__
        
        fname, slice = self.examples[i] 
        # Print statements 
        #print (fname,slice)
    
        with h5py.File(fname, 'r') as data:

            input_img  = data[self.key_img][slice,:,:,:]
            input_kspace  = data[self.key_kspace][slice,:,:,:]
            target = data['volfs'][slice,:,:,:]

            #input_img = np.transpose(input_img,[2,0,1])
            #input_kspace = np.transpose(input_kspace,[2,0,1])
            #target = np.transpose(target,[2,0,1])

            return torch.from_numpy(target),torch.from_numpy(input_img), torch.from_numpy(input_kspace), torch.from_numpy(input_kspace),torch.from_numpy(self.mask),torch.from_numpy(self.sensitivity),slice,str(fname.name)
 

