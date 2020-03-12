import h5py
import glob
import os
import numpy as np

h5_dir  = '/media/hticdeep/drive2/sriprabha/cardiac_dataset/validation'
save_path = '/media/hticdeep/drive2/sriprabha/cardiac_unet_results/reconstructions_zf/acc_4x'

#h5_dir  = '/media/hticdeep/drive2/sriprabha/kirby_dataset/h5_us_files/acc_4x/validation'
#save_path = '/media/hticdeep/drive2/sriprabha/kirby_unet_results/reconstructions_zf/acc_4x'

h5_files = glob.glob(os.path.join(h5_dir,'*.h5'))
#print (h5_files)

for h5_path in h5_files:

    filename = os.path.basename(h5_path)

    with h5py.File(h5_path,'r') as hf:
        #print(hf.keys())
        zf=hf['volus_4x'].value

    with h5py.File(os.path.join(save_path,filename),'w') as hf:

        zf = np.transpose(zf,[2,0,1])
        hf.create_dataset('reconstruction',data=zf)
