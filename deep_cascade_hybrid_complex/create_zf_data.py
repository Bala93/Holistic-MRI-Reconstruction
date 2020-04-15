from utils import complex_abs
import glob 
import os 
import h5py
import torch
from tqdm import tqdm


src_dir  = '/media/htic/NewVolume2/calgary_dataset/validation/acc_5x'
dst_dir  = '/media/htic/NewVolume1/murali/MR_reconstruction/experiments/calgary/acc_5x/zf/results'

src_files = glob.glob(os.path.join(src_dir,'*.h5'))

for h5_path in tqdm(src_files):

    with h5py.File(h5_path,'r') as hf:
        img_zf = torch.from_numpy(hf['img_volus_5x'].value)

    dst_path = os.path.join(dst_dir,os.path.basename(h5_path))

    with h5py.File(dst_path,'w') as hf:
        hf['reconstruction'] = complex_abs(img_zf)

    #break
