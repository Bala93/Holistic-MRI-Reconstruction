import pathlib
import random
import numpy as np
import h5py
from torch.utils.data import Dataset
import torch
from skimage import feature
import os 
import glob
from utils import npComplexToTorch


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

class SliceDataEvaluateMrbrain(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    #def __init__(self, root, acc_factor,dataset_type,mask_path): # acc_factor can be passed here and saved as self variable
    def __init__(self, root, acc_factor,dataset_type,predimgpath): # acc_factor can be passed here and saved as self variable
        # List the h5 files in root 
        files = list(pathlib.Path(root).iterdir())
        self.examples = []
        self.acc_factor = acc_factor 
        self.dataset_type = dataset_type
        self.key_img = 'img_volus_{}'.format(self.acc_factor)
        self.key_kspace = 'kspace_volus_{}'.format(self.acc_factor)
        self.reconspath = os.path.join(predimgpath,'results')
        #mask_path = os.path.join(mask_path,'mask_{}.npy'.format(acc_factor))
        #self.mask = np.load(mask_path)
        #print (files)
        self.flair_dir = root
        self.t1_dir    = root.replace('flair','t1')


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
        
        reconsfilename = os.path.join(self.reconspath,fname.name)
        t1_path = os.path.join(self.t1_dir,fname)

        # Print statements 
        #print (fname,slice)
        with h5py.File(reconsfilename,'r') as reconsdata:
            #print(reconsdata.keys())
            predictedimg = reconsdata['reconstruction'][slice,:,:]
    
        with h5py.File(fname, 'r') as data:

            input_img  = data[self.key_img][:,:,slice]
            input_kspace  = data[self.key_kspace][:,:,slice]
            input_kspace = npComplexToTorch(input_kspace)
    
            target = data['volfs'][:,:,slice].astype(np.float64)
            #print (input_img.shape,input_kspace.shape,target.shape)
        with h5py.File(t1_path, 'r') as data:

            t1_img = data['volfs'][:,:,slice].astype(np.float64)


            
            return torch.from_numpy(input_img), input_kspace, torch.from_numpy(target), torch.from_numpy(predictedimg),torch.from_numpy(t1_img)
 

class SliceDataDevMrbrain(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    #def __init__(self, root, acc_factor,dataset_type,mask_path): # acc_factor can be passed here and saved as self variable
    def __init__(self, root, acc_factor,predimgpath): # acc_factor can be passed here and saved as self variable
        # List the h5 files in root 
        files = glob.glob(os.path.join(root,'*.h5'))

        self.examples = []
        self.acc_factor = acc_factor 
        self.key_img = 'img_volus_{}'.format(self.acc_factor)
        self.key_kspace = 'kspace_volus_{}'.format(self.acc_factor)
        self.reconspath = os.path.join(predimgpath,'results')
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
        reconsfilename = os.path.join(self.reconspath,fname.name)

        with h5py.File(reconsfilename,'r') as reconsdata:
            #print(reconsdata.keys())
            predictedimg = reconsdata['reconstruction'][slice,:,:]
 
        with h5py.File(flair_path, 'r') as data:

            input_img  = data[self.key_img][:,:,slice]
            input_kspace  = data[self.key_kspace][:,:,slice]
            input_kspace = npComplexToTorch(input_kspace)
            target = data['volfs'][:,:,slice].astype(np.float64)

        with h5py.File(t1_path, 'r') as data:

            t1_img = data['volfs'][:,:,slice].astype(np.float64)

            return torch.from_numpy(input_img), input_kspace, torch.from_numpy(t1_img),torch.from_numpy(target),torch.from_numpy(predictedimg),slice,fname,torch.from_numpy(t1_img)



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

            input_img  = data[self.key_img][:,:,slice]
            input_kspace  = data[self.key_kspace][:,:,slice]
            input_kspace = npComplexToTorch(input_kspace)
    
            target = data['volfs'][:,:,slice].astype(np.float64)
            #print (input_img.shape,input_kspace.shape,target.shape)

            #kspace_cmplx = np.fft.fft2(target,norm='ortho')
            #uskspace_cmplx = kspace_cmplx * self.mask
            #zf_img = np.abs(np.fft.ifft2(uskspace_cmplx,norm='ortho'))
            
            #if self.dataset_type == 'cardiac':
                # Cardiac dataset should be padded,150 becomes 160. # this can be commented for kirby brain 
                #input_img  = np.pad(input_img,(5,5),'constant',constant_values=(0,0))
                #target = np.pad(target,(5,5),'constant',constant_values=(0,0))

            # Print statements
            #print (input.shape,target.shape)
            #return torch.from_numpy(zf_img), torch.from_numpy(target)
            # print (input_img.dtype,input_kspace.dtype,target.dtype)
            #print (torch.from_numpy(input_img), input_kspace, torch.from_numpy(target))
            return torch.from_numpy(input_img), input_kspace, torch.from_numpy(target)
 
class SliceDataEvaluate(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    #def __init__(self, root, acc_factor,dataset_type,mask_path): # acc_factor can be passed here and saved as self variable
    def __init__(self, root, acc_factor,dataset_type,predimgpath): # acc_factor can be passed here and saved as self variable
        # List the h5 files in root 
        files = list(pathlib.Path(root).iterdir())
        self.examples = []
        self.acc_factor = acc_factor 
        self.dataset_type = dataset_type
        self.key_img = 'img_volus_{}'.format(self.acc_factor)
        self.key_kspace = 'kspace_volus_{}'.format(self.acc_factor)
        self.reconspath = os.path.join(predimgpath,'results')
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
        
        reconsfilename = os.path.join(self.reconspath,fname.name)

        # Print statements 
        #print (fname,slice)
        with h5py.File(reconsfilename,'r') as reconsdata:
            #print(reconsdata.keys())
            predictedimg = reconsdata['reconstruction'][slice,:,:]
    
        with h5py.File(fname, 'r') as data:

            input_img  = data[self.key_img][:,:,slice]
            input_kspace  = data[self.key_kspace][:,:,slice]
            input_kspace = npComplexToTorch(input_kspace)
    
            target = data['volfs'][:,:,slice].astype(np.float64)
            #print (input_img.shape,input_kspace.shape,target.shape)

            #kspace_cmplx = np.fft.fft2(target,norm='ortho')
            #uskspace_cmplx = kspace_cmplx * self.mask
            #zf_img = np.abs(np.fft.ifft2(uskspace_cmplx,norm='ortho'))
            
            #if self.dataset_type == 'cardiac':
                # Cardiac dataset should be padded,150 becomes 160. # this can be commented for kirby brain 
                #input_img  = np.pad(input_img,(5,5),'constant',constant_values=(0,0))
                #target = np.pad(target,(5,5),'constant',constant_values=(0,0))

            # Print statements
            #print (input.shape,target.shape)
            #return torch.from_numpy(zf_img), torch.from_numpy(target)
            # print (input_img.dtype,input_kspace.dtype,target.dtype)
            #print (torch.from_numpy(input_img), input_kspace, torch.from_numpy(target))
            return torch.from_numpy(input_img), input_kspace, torch.from_numpy(target), torch.from_numpy(predictedimg)
            
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

            input_img  = data[self.key_img][:,:,slice]
            input_kspace  = data[self.key_kspace][:,:,slice]
            input_kspace = npComplexToTorch(input_kspace)
            target = data['volfs'][:,:,slice]

            #kspace_cmplx = np.fft.fft2(target,norm='ortho')
            #uskspace_cmplx = kspace_cmplx * self.mask
            #zf_img = np.abs(np.fft.ifft2(uskspace_cmplx,norm='ortho'))
 

            #if self.dataset_type == 'cardiac':
                # Cardiac dataset should be padded,150 becomes 160. # this can be commented for kirby brain 
                #input_img  = np.pad(input_img,(5,5),'constant',constant_values=(0,0))
                #target = np.pad(target,(5,5),'constant',constant_values=(0,0))

            # Print statements
            #print (input.shape,target.shape)
            return torch.from_numpy(input_img), input_kspace, torch.from_numpy(target),str(fname.name),slice
            #return torch.from_numpy(zf_img), torch.from_numpy(target),str(fname.name),slice


class SliceDataEvaluateDev(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    #def __init__(self, root, acc_factor,dataset_type,mask_path): # acc_factor can be passed here and saved as self variable
    def __init__(self, root, acc_factor,dataset_type,predimgpath): # acc_factor can be passed here and saved as self variable
        # List the h5 files in root 
        files = list(pathlib.Path(root).iterdir())
        self.examples = []
        self.acc_factor = acc_factor 
        self.dataset_type = dataset_type
        self.key_img = 'img_volus_{}'.format(self.acc_factor)
        self.key_kspace = 'kspace_volus_{}'.format(self.acc_factor)
        self.reconspath = os.path.join(predimgpath,'results')
        #mask_path = os.path.join(mask_path,'mask_{}.npy'.format(acc_factor))
        #self.mask = np.load(mask_path)
        #print (files)
        self.flair_dir = root
        self.t1_dir    = root.replace('flair','t1')


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
        
        reconsfilename = os.path.join(self.reconspath,fname.name)
        t1_path = os.path.join(self.t1_dir,fname)

        # Print statements 
        #print (fname,slice)
        with h5py.File(reconsfilename,'r') as reconsdata:
            #print(reconsdata.keys())
            predictedimg = reconsdata['reconstruction'][slice,:,:]
    
        with h5py.File(fname, 'r') as data:

            input_img  = data[self.key_img][:,:,slice]
            input_kspace  = data[self.key_kspace][:,:,slice]
            input_kspace = npComplexToTorch(input_kspace)
    
            target = data['volfs'][:,:,slice].astype(np.float64)
        with h5py.File(t1_path, 'r') as data:
            t1_img = data['volfs'][:,:,slice].astype(np.float64)


            #print (input_img.shape,input_kspace.shape,target.shape)

            #kspace_cmplx = np.fft.fft2(target,norm='ortho')
            #uskspace_cmplx = kspace_cmplx * self.mask
            #zf_img = np.abs(np.fft.ifft2(uskspace_cmplx,norm='ortho'))
            
            #if self.dataset_type == 'cardiac':
                # Cardiac dataset should be padded,150 becomes 160. # this can be commented for kirby brain 
                #input_img  = np.pad(input_img,(5,5),'constant',constant_values=(0,0))
                #target = np.pad(target,(5,5),'constant',constant_values=(0,0))

            # Print statements
            #print (input.shape,target.shape)
            #return torch.from_numpy(zf_img), torch.from_numpy(target)
            # print (input_img.dtype,input_kspace.dtype,target.dtype)
            #print (torch.from_numpy(input_img), input_kspace, torch.from_numpy(target))
            return torch.from_numpy(input_img), input_kspace, torch.from_numpy(target), torch.from_numpy(predictedimg),str(fname.name),slice,torch.from_numpy(t1_img)
 

