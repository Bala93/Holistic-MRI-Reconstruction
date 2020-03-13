import pathlib
import random
import numpy as np
import h5py
from torch.utils.data import Dataset
import torch
import os 


class KneeData(Dataset):

    def __init__(self, root): 

        files = list(pathlib.Path(root).iterdir())
        self.examples = []

        for fname in sorted(files):
            self.examples.append(fname)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        
        fname = self.examples[i] 

        with h5py.File(fname, 'r') as data:

            img_gt    = torch.from_numpy(data['img_gt'].value)
            img_und   = torch.from_numpy(data['img_und'].value)
            img_und_kspace = torch.from_numpy(data['img_und_kspace'].value)
            rawdata_und = torch.from_numpy(data['rawdata_und'].value)
            masks = torch.from_numpy(data['masks'].value)
            sensitivity = torch.from_numpy(data['sensitivity'].value)
            
            return img_gt,img_und,img_und_kspace,rawdata_und,masks,sensitivity


class KneeDataDev(Dataset):

    def __init__(self, root):

        files = list(pathlib.Path(root).iterdir())
        self.examples = []

        for fname in sorted(files):
            self.examples.append(fname) 

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        
        fname = self.examples[i]
    
        with h5py.File(fname, 'r') as data:

            img_gt    = torch.from_numpy(data['img_gt'].value)
            img_und   = torch.from_numpy(data['img_und'].value)
            img_und_kspace = torch.from_numpy(data['img_und_kspace'].value)
            rawdata_und = torch.from_numpy(data['rawdata_und'].value)
            masks = torch.from_numpy(data['masks'].value)
            sensitivity = torch.from_numpy(data['sensitivity'].value)
 
       
        return  img_gt,img_und,img_und_kspace,rawdata_und,masks,sensitivity,fname




