import pathlib
import random
import numpy as np
import h5py
from torch.utils.data import Dataset
import torch
from skimage import feature
import os 
from utils import npComplexToTorch
import glob



class SliceDataMrbrain(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    #def __init__(self, root, acc_factor,dataset_type,mask_path): # acc_factor can be passed here and saved as self variable
    def __init__(self, root, acc_factor): # acc_factor can be passed here and saved as self variable
        # List the h5 files in root 
        files = glob.glob(os.path.join(root,'*.h5'))

        self.examples = []
        self.acc_factor = acc_factor 
        self.key_img = 'img_volus_{}'.format(self.acc_factor)
        self.key_kspace = 'kspace_volus_{}'.format(self.acc_factor)
        #mask_path = os.path.join(mask_path,'mask_{}.npy'.format(acc_factor))
        #self.mask = np.load(mask_path)
        #print (files)

        self.flair_dir = root
        self.t1_dir    = root.replace('flair','t1')

        for file_path in sorted(files):
            with h5py.File(file_path,'r') as hf:
                #print (hf.keys())
                fsvol = hf['volfs']
                num_slices = fsvol.shape[2]
                self.examples += [(file_path, slice) for slice in range(num_slices)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # Index the fname and slice using the list created in __init__
        
        file_path, slice = self.examples[i] 
        fname = os.path.basename(file_path) 

        t1_path = os.path.join(self.t1_dir,fname)
        flair_path = os.path.join(self.flair_dir,fname)

        with h5py.File(flair_path, 'r') as data:

            input_img  = data[self.key_img][:,:,slice]
            input_kspace  = data[self.key_kspace][:,:,slice]
            input_kspace = npComplexToTorch(input_kspace)
            target = data['volfs'][:,:,slice].astype(np.float64)

        with h5py.File(t1_path, 'r') as data:

            t1_img = data['volfs'][:,:,slice].astype(np.float64)

            return torch.from_numpy(input_img), input_kspace, torch.from_numpy(t1_img),torch.from_numpy(target)


class SliceDataDevMrbrain(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    #def __init__(self, root, acc_factor,dataset_type,mask_path): # acc_factor can be passed here and saved as self variable
    def __init__(self, root, acc_factor): # acc_factor can be passed here and saved as self variable
        # List the h5 files in root 
        files = glob.glob(os.path.join(root,'*.h5'))

        self.examples = []
        self.acc_factor = acc_factor 
        self.key_img = 'img_volus_{}'.format(self.acc_factor)
        self.key_kspace = 'kspace_volus_{}'.format(self.acc_factor)
        #mask_path = os.path.join(mask_path,'mask_{}.npy'.format(acc_factor))
        #self.mask = np.load(mask_path)
        #print (files)

        self.flair_dir = root
        self.t1_dir    = root.replace('flair','t1')

        for file_path in sorted(files):
            with h5py.File(file_path,'r') as hf:
                #print (hf.keys())
                fsvol = hf['volfs']
                num_slices = fsvol.shape[2]
                self.examples += [(file_path, slice) for slice in range(num_slices)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # Index the fname and slice using the list created in __init__
        
        file_path, slice = self.examples[i] 
        fname = os.path.basename(file_path) 

        t1_path = os.path.join(self.t1_dir,fname)
        flair_path = os.path.join(self.flair_dir,fname)

        with h5py.File(flair_path, 'r') as data:

            input_img  = data[self.key_img][:,:,slice]
            input_kspace  = data[self.key_kspace][:,:,slice]
            input_kspace = npComplexToTorch(input_kspace)
            target = data['volfs'][:,:,slice].astype(np.float64)

        with h5py.File(t1_path, 'r') as data:

            t1_img = data['volfs'][:,:,slice].astype(np.float64)

            return torch.from_numpy(input_img), input_kspace, torch.from_numpy(t1_img),torch.from_numpy(target),slice,fname










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
            t2kspaceus= npComplexToTorch(t2kspaceus)
    
            return torch.from_numpy(t2imgus), t2kspaceus, torch.from_numpy(t1imgfs), torch.from_numpy(t2imgfs),slice,str(fname.name)
 
