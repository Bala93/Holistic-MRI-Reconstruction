import bart
import h5py
import numpy as np
import os 
from tqdm import tqdm
import pandas as pd



root = '/media/htic/hd/multicoil_knee_fastmri/dataset/multicoil_val'
savedir = '/media/htic/hd/multicoil_knee_fastmri/sensitivity/valid'
csvpath = '/media/htic/NewVolume3/Balamurali/fastmri/invalid_h5_valid.csv'

#root = '/media/htic/NewVolume5/fastmri/multicoil/multicoil_train'
#savedir = '/media/htic/hd/multicoil_knee_fastmri/sensitivity/train'
#csvpath = '/media/htic/NewVolume3/Balamurali/fastmri/invalid_h5_train.csv'

#root = '/media/htic/NewVolume3/Balamurali/fastmri/dataset'
#savedir = '/media/htic/NewVolume3/Balamurali/fastmri/sensitivity'
#csvpath = '/media/htic/NewVolume3/Balamurali/fastmri/invalid_h5_train_vol3.csv'

invalid_h5_info = {'filename':[]}

for path, subdirs, files in os.walk(root):
  
    for name in tqdm(files):

        h5path = os.path.join(path, name)
        volname = path.split('/')[-1]
        savevol = os.path.join(savedir, volname)
        savefile = os.path.join(savevol, name)

        if not os.path.exists(savevol):
            os.mkdir(savevol)

        if os.path.exists(savefile):
            continue 

        #print (h5path, savefile)

        try: 

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
    
            with h5py.File(savefile, 'w') as hf1:
    
                hf1['sensitivity'] = sens_maps

        except: 
            print (h5path)
            invalid_h5_info['filename'].append(h5path)
    
        #break

    #break

df = pd.DataFrame(invalid_h5_info)
df.to_csv(csvpath)

