import h5py
import torch
import numpy as np



def save_reconstructions(reconstructions, out_dir):

    for fname, recons in reconstructions.items():

        with h5py.File(str(out_dir) +'/' +str(fname), 'w') as f:
            f.create_dataset('reconstruction', data=recons)

def npComplexToTorch(kspace_np):

    # Converts a numpy complex to torch 
    kspace_real_torch=torch.from_numpy(kspace_np.real)
    kspace_imag_torch=torch.from_numpy(kspace_np.imag)
    kspace_torch = torch.stack([kspace_real_torch,kspace_imag_torch],dim=2)
    
    return kspace_torch
