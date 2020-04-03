
import h5py
import glob
import os
import numpy as np
from argparse import ArgumentParser
import argparse
import bart 
from tqdm import tqdm 
from skimage import measure

def create_tv_data(h5_dir,save_path,acc_factor):

    h5_files = glob.glob(os.path.join(h5_dir,'*.h5'))
    print ("Number of files:",len(h5_files))
    
    for h5_path in tqdm(h5_files):

        filename = os.path.basename(h5_path)

        with h5py.File(h5_path,'r') as hf:

            img_volfs    = hf['volfs'].value
            img_volus    = hf['img_volus_{}'.format(acc_factor)].value
            kspace_volus = hf['kspace_volus_{}'.format(acc_factor)].value

            if True:

                img_volfs    = img_volfs[:,:,:,0] + 1j * img_volfs[:,:,:,1]
                img_volus    = img_volus[:,:,:,0] + 1j * img_volus[:,:,:,1]
                kspace_volus = kspace_volus[:,:,:,0] + 1j * kspace_volus[:,:,:,1]

                img_volfs = np.transpose(img_volfs,(1,2,0))
                img_volus = np.transpose(img_volus,(1,2,0))
                kspace_volus = np.transpose(kspace_volus,(1,2,0))

            kspace_volfs = np.fft.fft2(img_volfs,norm='ortho',axes=(0,1))

        recon_list = []

        print (img_volfs.shape,kspace_volfs.shape,img_volus.shape,kspace_volus.shape)

        for ii in tqdm(range(img_volfs.shape[-1])):

            #img_slicefs    = img_volfs[:,:,ii]
            #img_sliceus    = img_volus[:,:,ii]

            kspace_slicefs = kspace_volfs[:,:,ii]
            kspace_sliceus = kspace_volus[:,:,ii]
            
            sens_maps = bart.bart(1,'ecalib -d0 -m1',kspace_slicefs) 
            recons = bart.bart(1,'pics -d0 -S -R T:7:0:0.01 -i 100',kspace_sliceus,sens_maps)
            recons = np.fft.ifftshift(recons)
            recons = np.abs(recons)
            recon_list.append(recons)

            #print (measure.compare_psnr(recons,img_slicefs,np.max(img_slicefs) - np.min(img_slicefs)))
            #print (measure.compare_psnr(img_sliceus,img_slicefs,np.max(img_slicefs) - np.min(img_slicefs)))

        recon_vol = np.stack(recon_list,axis=0)

        #print (recon_vol.shape)

        with h5py.File(os.path.join(save_path,filename),'w') as hf:
            hf.create_dataset('reconstruction',data=recon_vol)

        #break

    return 

if __name__ == '__main__':

    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--validation_path', type=str, required=True,
                        help='Path to validation data')
    parser.add_argument('--tv_save_path', type=str, required=True,
                        help='Path to save zf data')
    parser.add_argument('--acc_factor', type=str, required=True,
                        help='acc factor')

    args = parser.parse_args()

    create_tv_data(args.validation_path,args.tv_save_path,args.acc_factor)


