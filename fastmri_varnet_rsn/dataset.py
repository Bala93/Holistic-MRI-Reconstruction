"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import pathlib
import random

import numpy as np
import h5py
from torch.utils.data import Dataset
import pandas as pd


class SliceData(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root, root_csv, transform, challenge, sample_rate=1):
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


        df  = pd.read_csv(root_csv)

        fnames = df['filename']
        slices = df['sliceno']

        files = []

        for fname, slice in zip(fnames, slices):

            files += [root / '{}'.format(fname) / '{}'.format(slice)] 

        #files = list(pathlib.Path(root).iterdir())
        if sample_rate < 1:
            random.shuffle(files)
            num_files = round(len(files) * sample_rate)
            files = files[:num_files]

        self.examples = files

        #print (len(self.examples))

        '''
        for fname in sorted(files):
            data = h5py.File(fname, 'r')

            # Compute the size of zero padding in k-space
            # We really should have stored this as an attribute in the hdf5 file
            try:
                import ismrmrd
                hdr = ismrmrd.xsd.CreateFromDocument(
                    data['ismrmrd_header'][()])
                enc = hdr.encoding[0]
                enc_size = (enc.encodedSpace.matrixSize.x,
                            enc.encodedSpace.matrixSize.y,
                            enc.encodedSpace.matrixSize.z)
                enc_limits_center = enc.encodingLimits.kspace_encoding_step_1.center
                enc_limits_max = enc.encodingLimits.kspace_encoding_step_1.maximum + 1
                padding_left = enc_size[1] // 2 - enc_limits_center
                padding_right = padding_left + enc_limits_max
            except Exception as e:
                padding_left = None
                padding_right = None
                raise e

            kspace = data['kspace']
            num_slices = kspace.shape[0]
            self.examples += [(fname, slice, padding_left, padding_right) for slice in range(num_slices)]
        '''

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        #fname, slice, padding_left, padding_right = self.examples[i]
        fname = self.examples[i]
        with h5py.File(fname, 'r') as data:
            #kspace = data['kspace'][slice]
            kspace = data['kspace'].value
            mask = np.asarray(data['mask'].value) if 'mask' in data else None
            target = data[self.recons_key].value if self.recons_key in data else None
            attrs = dict(data.attrs)
            attrs['max'] = target.max() if self.recons_key in data else None
            #attrs['padding_left'] = padding_left
            attrs['padding_left'] = 0
            #attrs['padding_right'] = padding_right
            attrs['padding_right'] = kspace.shape[-1]
            return self.transform(kspace, mask, target, attrs, str(fname))
