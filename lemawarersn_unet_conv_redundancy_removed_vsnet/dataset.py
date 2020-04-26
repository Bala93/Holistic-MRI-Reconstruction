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

            img_gt    = torch.from_numpy(data['img_gt'].value) #h,w,2
            img_und   = torch.from_numpy(data['img_und'].value)#h,w,2
            img_und_kspace = torch.from_numpy(data['img_und_kspace'].value)#h,w,2
            rawdata_und = torch.from_numpy(data['rawdata_und'].value)#15,h,w,2
            masks = torch.from_numpy(data['masks'].value)#15,h,w,2
            sensitivity = torch.from_numpy(data['sensitivity'].value)#15,hw,2
            
            return img_gt,img_und,img_und_kspace,rawdata_und,masks,sensitivity


class KneeDataEvaluate(Dataset):

    def __init__(self, root,predimgpath): 

        files = list(pathlib.Path(root).iterdir())
        self.examples = []
        self.reconspath = os.path.join(predimgpath,'results')

        for fname in sorted(files):
            self.examples.append(fname)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        
        fname = self.examples[i] 
        reconsfilename = os.path.join(self.reconspath,fname.name)

        with h5py.File(reconsfilename,'r') as reconsdata:
            predictedimg = torch.from_numpy(reconsdata['reconstruction'].value) #1,h,
        with h5py.File(fname, 'r') as data:

            img_gt    = torch.from_numpy(data['img_gt'].value) #h,w,2
            img_und   = torch.from_numpy(data['img_und'].value)#h,w,2
            img_und_kspace = torch.from_numpy(data['img_und_kspace'].value)#h,w,2
            rawdata_und = torch.from_numpy(data['rawdata_und'].value)#15,h,w,2
            masks = torch.from_numpy(data['masks'].value)#15,h,w,2
            sensitivity = torch.from_numpy(data['sensitivity'].value)#15,hw,2
            
            return img_gt,img_und,img_und_kspace,rawdata_und,masks,sensitivity,predictedimg


class KneeDataDevEvaluate(Dataset):

    def __init__(self, root, predimgpath): 

        files = list(pathlib.Path(root).iterdir())
        self.examples = []
        self.reconspath = os.path.join(predimgpath,'results')

        for fname in sorted(files):
            self.examples.append(fname)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        
        fname = self.examples[i] 
        reconsfilename = os.path.join(self.reconspath,fname.name)

        with h5py.File(reconsfilename,'r') as reconsdata:
            predictedimg = torch.from_numpy(reconsdata['reconstruction'].value) #1,h,
        with h5py.File(fname, 'r') as data:

            img_gt    = torch.from_numpy(data['img_gt'].value) #h,w,2
            img_und   = torch.from_numpy(data['img_und'].value)#h,w,2
            img_und_kspace = torch.from_numpy(data['img_und_kspace'].value)#h,w,2
            rawdata_und = torch.from_numpy(data['rawdata_und'].value)#15,h,w,2
            masks = torch.from_numpy(data['masks'].value)#15,h,w,2
            sensitivity = torch.from_numpy(data['sensitivity'].value)#15,hw,2
            
            return img_gt,img_und,img_und_kspace,rawdata_und,masks,sensitivity,predictedimg,str(fname.name)



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
 
       
        return  img_gt,img_und,img_und_kspace,rawdata_und,masks,sensitivity,str(fname.name)
