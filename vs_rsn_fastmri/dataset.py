import pathlib
import random
import numpy as np
import h5py
from torch.utils.data import Dataset
import torch
import os 
import pandas as pd

class SliceData(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self,root, csv_path, transform, challenge='singlecoil', sample_rate=1):
        """
        Args:
            root (pathlib.Path): Path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.
            challenge (str): "singlecoil" or "multicoil" depending on which challenge to use.
            sample_rate (float, optional): A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.
        """
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        print (challenge)
        self.recons_key = 'reconstruction_esc' if challenge == 'singlecoil' else 'reconstruction_rss'

        dataset_path = os.path.join(root, 'dataset')
        sensitivity_path = os.path.join(root, 'sensitivity')

        self.transform = transform

        self.examples = []

        df = pd.read_csv(csv_path)

        files = df['filename']
        slices = df['sliceno']

        print ("Preparing data")

        for fname, slice in zip(files, slices):

            self.examples += [(os.path.join(dataset_path, fname,slice), os.path.join(sensitivity_path, fname, slice))] 

        if sample_rate < 1:

            #random.shuffle(self.examples)
            num_files = round(len(self.examples) * sample_rate)
            self.examples= self.examples[:num_files]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):

        data_fname,sensitivity_fname  =  self.examples[i]
        #print (data_fname, sensitivity_fname, self.recons_key)

        with h5py.File(data_fname, 'r') as data:

            kspace = data['kspace'].value
            target = data[self.recons_key].value
            mask = np.asarray(data['mask']) if 'mask' in data else None # train, valid will return None

        with h5py.File(sensitivity_fname, 'r') as data:

            sensitivity = data['sensitivity'].value

        return self.transform(kspace, sensitivity, target, mask, data_fname, sensitivity_fname)


class SliceDataDev(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root, transform, challenge='singlecoil', sample_rate=1):
        """
        Args:
            root (pathlib.Path): Path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.
            challenge (str): "singlecoil" or "multicoil" depending on which challenge to use.
            sample_rate (float, optional): A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.
        """
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        self.transform = transform
        self.recons_key = 'reconstruction_esc' if challenge == 'singlecoil' \
            else 'reconstruction_rss'

        self.examples = []

        df = pd.read_csv(csv_path)
        files = df['fname']
        slices = df['slice']

        print ("Preparing data")

        for fname, slice in zip(files, slices):
            self.examples += [(fname, slice, 0, 0)]

        print ("Validation preparation done")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice, padding_left, padding_right = self.examples[i]
        with h5py.File(fname, 'r') as data:
            kspace = data['kspace'][slice]
            mask = np.asarray(data['mask']) if 'mask' in data else None
            target = data[self.recons_key][slice] if self.recons_key in data else None
            attrs = dict(data.attrs)
            attrs['padding_left'] = padding_left
            attrs['padding_right'] = padding_right
            return self.transform(kspace, mask, target, attrs, fname, slice)


