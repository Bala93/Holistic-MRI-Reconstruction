import h5py                                                           
import glob                                                         
import os                                                            
from tqdm import tqdm


## Code to convert the volume h5 to slice h5

base_path = '/media/htic/hd/multicoil_train/'
save_path = '/media/htic/NewVolume5/fastmri/multicoil/multicoil_train'
files = glob.glob(base_path + '*.h5')                                 
slicecount = 0                                                        

for filepath in tqdm(files):     

    fname = os.path.basename(filepath)                               
    save_dir = os.path.join(save_path, fname[:-3])                    

    if not os.path.exists(save_dir):                               
        os.mkdir(save_dir)                                            

    with h5py.File(filepath, 'r') as hf:                            
        kspace = hf['kspace'].value                                  
        target = hf['reconstruction_rss'].value                       

    for kk in range(kspace.shape[0]):

        savefile = os.path.join(save_dir, '{}.h5'.format(kk))         
        ks, tar = kspace[kk, :, :], target[kk, :, :]                  

        with h5py.File(savefile, 'w') as hf:                          

            hf['kspace'] = ks                                        
            hf['reconstruction_rss'] = tar                          
    #break
