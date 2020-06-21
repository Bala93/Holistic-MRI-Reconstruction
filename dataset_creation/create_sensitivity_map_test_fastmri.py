import bart
import h5py
import numpy as np
import os 
from tqdm import tqdm
import pandas as pd
import glob 

root = '/media/htic/hd/fastmri/multicoil_test_v2/dataset/'
savedir = '/media/htic/hd/fastmri/multicoil_test_v2/sensitivity/'

files = glob.glob(root + '*.h5')
  
for h5path in tqdm(files):

    name = os.path.basename(h5path)
    savefile = os.path.join(savedir, name)

    with h5py.File(h5path, 'r') as hf:

        vol_kspace = hf['kspace'].value 

    sens_map_list = []
  
    for kk in tqdm(range(vol_kspace.shape[0])):

        kspace = vol_kspace[kk, :, :, :] 

        img = np.fft.ifftshift(np.fft.ifft2(kspace,norm='ortho'),axes=(1,2))
        img_abs = np.abs(img)
        img_abs_max = np.max(img_abs)
        
        kspace = np.transpose(kspace, (1, 2, 0))
        kspace = kspace[:,:,None,:] / img_abs_max
        
        sens_map = bart.bart(1, 'ecalib -d0 -m1 -r30', kspace)[:,:,0,:]
        sens_map = np.transpose(sens_map, [2, 0, 1])

        sens_map_list.append(sens_map)

    sens_maps = np.stack(sens_map_list, axis=0)

    with h5py.File(savefile, 'w') as hf1:

        hf1['sensitivity'] = sens_maps

    #break
