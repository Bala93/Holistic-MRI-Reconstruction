import os 
import numpy as np
import h5py
import pandas as pd 

csv_info = {'filename':[],'sliceno':[]}
savepath = '/media/htic/NewVolume3/Balamurali/fastmri/multicoil_train.csv'
root = ''

for path, subdirs, files in os.walk(root):

    volname = path.split('/')[-1]

    for name in tqdm(files):

        h5path = os.path.join(path, name)

        with h5py.File(h5path, 'r') as hf: 

            img = hf['reconstruction_rss'].value
       
        for sl in range(img.shape[0]):

            csv_info['filename'].append(volname)
            csv_info['sliceno'].append(sl)

    break

df = pd.DataFrame(csv_info)
df.to_csv(save_path)
