from tqdm import tqdm 
import glob
import os 
import h5py
from utils import undersample_volume_static,normalize_volume
import numpy as np 


# The codes helps to create h5 files with fs and mentioned us acceleration factors. 
# TODO: Number of lines in the center

'''
# kirby dataset
dataset_type = 'kirby'
src_path = '/media/htic/NewVolume1/murali/MR_reconstruction/kirby_recon_dataset/validation/*.h5'
dst_dir  = '/media/htic/NewVolume1/murali/MR_reconstruction/datasets/{}/validation'.format(dataset_type)
mask_path = '/media/htic/NewVolume1/murali/MR_reconstruction/Reconstruction-for-MRI/us_masks/{}'.format(dataset_type) # Prepare a mask here with image dimension 

# cardiac dataset
dataset_type = 'cardiac'
src_path = '/media/htic/NewVolume3/Balamurali/mri_recon_local/training/*.h5' # change both this lines before running 
dst_dir  = '/media/htic/NewVolume1/murali/MR_reconstruction/datasets/cardiac/train' # change both this lines before running 
mask_path = '/media/htic/NewVolume1/murali/MR_reconstruction/Reconstruction-for-MRI/us_masks/{}'.format(dataset_type) # Prepare a mask here with image dimension 
'''

'''
dataset_type = 'mrbrain'
src_path = '/media/htic/NewVolume3/Balamurali/mrbrains_dataset/validation/*.h5' # change both this lines before running 
dst_dir  = '/media/htic/NewVolume1/murali/MR_reconstruction/datasets/mrbrain/validation' # change both this lines before running 
mask_path = '/media/htic/NewVolume1/murali/MR_reconstruction/Reconstruction-for-MRI/us_masks/{}/cartesian'.format(dataset_type) # Prepare a mask here with image dimension 
'''

#dataset_type = 'mrbrain'
#src_path = '/media/htic/NewVolume3/Balamurali/mrbrains_dataset_flair/valid/*.h5' # change both this lines before running 
#dst_dir  = '/media/htic/NewVolume1/murali/MR_reconstruction/datasets/mrbrain/validation' # change both this lines before running 
#mask_path = '/media/htic/NewVolume1/murali/MR_reconstruction/Reconstruction-for-MRI/us_masks/{}'.format(dataset_type) # Prepare a mask here with image dimension 



'''
dataset_type = 'cardiac'
#mask_type = 'gaussian'
mask_type = 'cartesian'
src_path = '/media/htic/NewVolume3/Balamurali/mri_recon_local/{}/training/*.h5'.format(dataset_type) # change both this lines before running 
dst_dir  = '/media/htic/NewVolume1/murali/MR_reconstruction/datasets/cardiac/{}/train'.format(mask_type) # change both this lines before running 
mask_path = '/media/htic/NewVolume1/murali/MR_reconstruction/Reconstruction-for-MRI/us_masks/{}/{}'.format(dataset_type,mask_type) # Prepare a mask here with image dimension 
#mask_path = '/media/htic/NewVolume1/murali/MR_reconstruction/Reconstruction-for-MRI/us_masks/{}'.format(dataset_type)

'''

# kirby small dataset
'''
dataset_type = 'kirby_small'
src_path = '/media/htic/NewVolume5/mr_recon_backup/MR_reconstruction/datasets/kirby_small/cartesian/validation/acc_2x/*.h5' # take the volfs from cartesian data and use it to produce gaussian us data for kirby_small
dst_dir  = '/media/htic/NewVolume5/mr_recon_backup/MR_reconstruction/datasets/{}/gaussian/validation'.format(dataset_type)
mask_path = '/media/htic/NewVolume1/murali/MR_reconstruction/Reconstruction-for-MRI/us_masks/{}/gaussian'.format(dataset_type) # Prepare a mask here with image dimension 
'''

'''
dataset_type = 'calgary'
src_path = '/media/htic/NewVolume3/Balamurali/calgary_dataset/val/*.h5' # take the volfs from cartesian data and use it to produce gaussian us data for kirby_small
dst_dir  = '/media/htic/NewVolume3/Balamurali/calgary_dataset/' #.format(dataset_type)
mask_path = '/media/htic/NewVolume1/murali/MR_reconstruction/Reconstruction-for-MRI/us_masks/{}/'.format(dataset_type) # Prepare a mask here with image dimension 
'''


dataset_type = 'knee_mrnet'
src_path = '/media/htic/NewVolume3/Balamurali/knee_mri_mrnet/valid/*.h5' 
dst_dir  = '/media/htic/NewVolume3/Balamurali/knee_mri_mrnet_dataset/cartesian/validation/' 
mask_path = '/media/htic/NewVolume1/murali/MR_reconstruction/Reconstruction-for-MRI/us_masks/{}/cartesian/'.format(dataset_type) # Prepare a mask here with image dimension 


#us_factors     = [2,2.5,3.3,4,5,8]
us_factors     = [5]
#us_factors     = [3,4.5,6]
#us_factors     = [2.5]
us_str_factors = [str(ii).replace('.','_') for ii in us_factors]

us_masks_path = [os.path.join(mask_path,'mask_{}x.npy'.format(ii)) for ii in us_str_factors]
#us_masks = [np.fft.fftshift(np.load(us_mask_path)) for us_mask_path in us_masks_path]
print (us_masks_path)
us_masks = [np.load(us_mask_path) for us_mask_path in us_masks_path]

for iter,path in enumerate(tqdm(glob.glob(src_path))):

    
    with h5py.File(path,'r') as hf1:
        #img_vol = hf1['inpVol']
        #norm_img_vol = normalize_volume(img_vol)
        norm_img_vol  = hf1['volfs'].value
        #norm_img_vol  = norm_img_vol[:,:,50:120]
        #norm_img_vol  = norm_img_vol[:,:,50:120]
        print(norm_img_vol.shape) 
        #mask_vol = hf1['mask'].value 
        us_vol = undersample_volume_static(norm_img_vol,us_str_factors,us_masks)
   

    for us_factor in tqdm(us_str_factors): 

        dst_save_dir = os.path.join(dst_dir,'acc_{}x'.format(us_factor))  

        if not os.path.exists(dst_save_dir):
            os.mkdir(dst_save_dir)        

        dst_save_path = os.path.join(dst_save_dir,'{}.h5'.format(iter))
        print(dst_save_dir) 

        with h5py.File(dst_save_path,'w') as hf2:

            hf2.create_dataset('volfs',data=norm_img_vol)   
            #hf2.create_dataset('mask',data=mask_vol)   

            vol_us_name = 'img_volus_{}x'.format(us_factor)
            hf2.create_dataset(vol_us_name,data=us_vol[vol_us_name])

            vol_us_name = 'kspace_volus_{}x'.format(us_factor)
            hf2.create_dataset(vol_us_name,data=us_vol[vol_us_name])
    
       #break


    #break
