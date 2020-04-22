import pathlib
import random
import numpy as np
import h5py
from torch.utils.data import Dataset
import torch
from skimage import feature
import os 
from utils import npComplexToTorch


class SliceData(Dataset):

    def __init__(self, root,dataset_type): # acc_factor can be passed here and saved as self variable
        # List the h5 files in root 
        files = list(pathlib.Path(root).iterdir())
        self.examples = []
        self.dataset_type = dataset_type

        for fname in sorted(files):
            with h5py.File(fname,'r') as hf:
                fsvol = hf['t1imgfs']
                num_slices = fsvol.shape[2]
                self.examples += [(fname, slice) for slice in range(num_slices)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        
        fname, slice = self.examples[i] 
    
        with h5py.File(fname, 'r') as data:

            t1imgfs  = data['t1imgfs'][:,:,slice]
            t2imgfs  = data['t2imgfs'][:,:,slice]
            t2imgus  = data['t2imgus'][:,:,slice]

            t2kspaceus  = data['t2kspaceus'][:,:,slice]
            t2kspaceus= npComplexToTorch(t2kspaceus)
    
            return torch.from_numpy(t2imgus), t2kspaceus, torch.from_numpy(t1imgfs), torch.from_numpy(t2imgfs)
            
class SliceDataDev(Dataset):

    def __init__(self, root,dataset_type):

        # List the h5 files in root 
        files = list(pathlib.Path(root).iterdir())
        self.examples = []
        self.dataset_type = dataset_type

        for fname in sorted(files):
            with h5py.File(fname,'r') as hf:
                fsvol = hf['t1imgfs']
                num_slices = fsvol.shape[2]
                self.examples += [(fname, slice) for slice in range(num_slices)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        
        fname, slice = self.examples[i] 
    
        with h5py.File(fname, 'r') as data:

            t1imgfs  = data['t1imgfs'][:,:,slice]
            t2imgfs  = data['t2imgfs'][:,:,slice]
            t2imgus  = data['t2imgus'][:,:,slice]

            t2kspaceus  = data['t2kspaceus'][:,:,slice]
            t2kspaceus= npComplexToTorch(input_kspace)
    
            return torch.from_numpy(t2imgus), t2kspaceus, torch.from_numpy(t1imgfs), torch.from_numpy(t2imgfs)
 
