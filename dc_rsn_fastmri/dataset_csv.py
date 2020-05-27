"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

#import pathlib
import random

import numpy as np
import h5py
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd 
import os 


class SliceData(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root, csv_path, transform, challenge='singlecoil', sample_rate=1):
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

        print (csv_path)

        df = pd.read_csv(csv_path)
        files = df['fname']
        slices = df['slice']

        print ("Preparing data")
        #print (root)

        for fname, slice in zip(files, slices):

            self.examples += [(os.path.join(root, fname), slice, 0, 0)] ## 0, 0 is to compensate for padding_left and padding_right 

        print ("Train preparation done")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice, padding_left, padding_right = self.examples[i]
        with h5py.File(fname, 'r') as data:
            kspace = data['kspace'][slice]
            mask = np.asarray(data['mask']) if 'mask' in data else None
            target = data[self.recons_key][slice] if self.recons_key in data else None
            attrs = dict(data.attrs)
            attrs['padding_left'] = padding_left # not used in transform
            attrs['padding_right'] = padding_right # not used in transform
            return self.transform(kspace, mask, target, attrs, fname, slice)


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

