from data import transforms as T
import glob 
import os 
import h5py
import torch
from tqdm import tqdm


src_dir  = '/media/htic/NewVolume5/knee_mri_vsnet_globus/axial_t2_h5/validation'
dst_dir  = '/media/htic/NewVolume5/knee_mri_vsnet_globus/axial_t2_h5/zf/results'

src_files = glob.glob(os.path.join(src_dir,'*.h5'))

for h5_path in tqdm(src_files):

    with h5py.File(h5_path,'r') as hf:

        img_zf = torch.from_numpy(hf['img_und'].value)
        img_zf = img_zf.unsqueeze(0)

    dst_path = os.path.join(dst_dir,os.path.basename(h5_path))

    with h5py.File(dst_path,'w') as hf:

        hf['reconstruction'] = T.complex_abs(img_zf)

    #break
