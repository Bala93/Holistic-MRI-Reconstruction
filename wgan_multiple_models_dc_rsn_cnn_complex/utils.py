import h5py
import torch
import numpy as np



def save_reconstructions(reconstructions, out_dir):

    out_dir.mkdir(exist_ok=True)

    for fname, recons in reconstructions.items():

        with h5py.File(str(out_dir) +'/' +str(fname), 'w') as f:
            f.create_dataset('reconstruction', data=recons)

def npComplexToTorch(kspace_np):

    # Converts a numpy complex to torch 
    kspace_real_torch=torch.from_numpy(kspace_np.real)
    kspace_imag_torch=torch.from_numpy(kspace_np.imag)
    kspace_torch = torch.stack([kspace_real_torch,kspace_imag_torch],dim=2)
    
    return kspace_torch

def complex_abs(data):
    """
    Compute the absolute value of a complex valued input tensor.

    Args:
        data (torch.Tensor): A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        torch.Tensor: Absolute value of data
    """
    assert data.size(-1) == 2
    return (data ** 2).sum(dim=-1).sqrt()

