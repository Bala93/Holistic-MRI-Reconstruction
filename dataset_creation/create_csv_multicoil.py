import os 
import numpy as np
import h5py
import pandas as pd 
from tqdm import tqdm

csv_info = {'filename':[],'sliceno':[]}
root = '/media/htic/NewVolume5/fastmri/multicoil_train/dataset'
savepath = '/media/htic/NewVolume3/Balamurali/fastmri/multicoil_train.csv'
#root = '/media/htic/hd/fastmri/multicoil_valid/dataset'
#savepath = '/media/htic/NewVolume3/Balamurali/fastmri/multicoil_valid.csv'

for path, subdirs, files in os.walk(root):

    for name in tqdm(files):

        h5path = os.path.join(path, name)
        h5path_list = h5path.split('/')
     
        volname = h5path_list[-2]
        slname = h5path_list[-1]        

        try:

            with h5py.File(h5path, 'r') as hf: 
    
                img = hf['reconstruction_rss'].value
           
            csv_info['filename'].append(volname)
            csv_info['sliceno'].append(slname)
    
        except:
            continue


df = pd.DataFrame(csv_info)
df.to_csv(savepath)
