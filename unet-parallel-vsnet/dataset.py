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

            img_gt = img_gt.permute(2,0,1)
            img_und = img_und.permute(2,0,1)
            
            return img_gt,img_und


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


            img_gt = img_gt.permute(2,0,1)
            img_und = img_und.permute(2,0,1)
       
        return  img_gt,img_und




