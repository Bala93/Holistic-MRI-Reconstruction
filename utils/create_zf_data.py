import h5py
import glob
import os
import numpy as np
from argparse import ArgumentParser
import argparse

def create_zf_data(h5_dir,save_path,acc_factor):

    h5_files = glob.glob(os.path.join(h5_dir,'*.h5'))
    print ("Number of files:",len(h5_files))
    
    for h5_path in h5_files:

        filename = os.path.basename(h5_path)

        with h5py.File(h5_path,'r') as hf:
            zf=hf['img_volus_{}'.format(acc_factor)].value

        print (zf.shape)

        with h5py.File(os.path.join(save_path,filename),'w') as hf:
            zf = np.transpose(zf,[2,0,1])
            hf.create_dataset('reconstruction',data=zf)

    return 

if __name__ == '__main__':

    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--validation_path', type=str, required=True,
                        help='Path to validation data')
    parser.add_argument('--zf_save_path', type=str, required=True,
                        help='Path to save zf data')
    parser.add_argument('--acc_factor', type=str, required=True,
                        help='acc factor')
    parser.add_argument('--dataset_type', type=str, required=True,help='dataset_type')

    args = parser.parse_args()

    #create_zf_data(args.validation_path,args.zf_save_path,args.acc_factor,args.dataset_type)
    create_zf_data(args.validation_path,args.zf_save_path,args.acc_factor)


