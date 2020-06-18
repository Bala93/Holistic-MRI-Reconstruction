import bart
import h5py
import numpy as np
import os 
from tqdm import tqdm


root = '/media/htic/NewVolume5/fastmri/multicoil/multicoil_train'
savedir = '/media/htic/NewVolume5/fastmri/multicoil/sensitivity_train'

for path, subdirs, files in os.walk(root):

    for name in files:

        h5path = os.path.join(path, name)
        volname = path.split('/')[-1]

        with h5py.File(h5path, 'r') as hf:

            kspace = hf['kspace'].value 
            img = np.fft.ifftshift(np.fft.ifft2(kspace,norm='ortho'),axes=(1,2))
            
            img_abs = np.abs(img)
            img_abs_max = np.max(img_abs)
            
            img = np.transpose(img, (1, 2, 0))
            img = img[:,:,None,:] / img_abs_max
            
            kspace = np.transpose(kspace, (1, 2, 0))
            kspace = kspace[:,:,None,:] / img_abs_max
            
            sens_maps = bart.bart(1, 'ecalib -d0 -m1 -r30', kspace)[:,:,0,:]
            sens_maps = np.transpose(sens_maps, [2, 0, 1])

            savevol = os.path.join(savedir, volname)
          
            if not os.path.exists(savevol):
                os.mkdir(savevol)

            savefile = os.path.join(savevol, name)

        with h5py.File(savefile, 'w') as hf1:

            hf1['sensitivity'] = sens_maps

        #break

    #break

