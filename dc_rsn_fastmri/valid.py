import pathlib
import sys
from collections import defaultdict
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import SliceDataDev
from models import DnCn
import h5py
from tqdm import tqdm
import transforms as T


class DataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self, resolution, which_challenge, mask_func=None, use_seed=True):
        """
        Args:
            mask_func (common.subsample.MaskFunc): A function that can create a mask of
                appropriate shape.
            resolution (int): Resolution of the image.
            which_challenge (str): Either "singlecoil" or "multicoil" denoting the dataset.
            use_seed (bool): If true, this class computes a pseudo random number generator seed
                from the filename. This ensures that the same mask is used for all the slices of
                a given volume every time.
        """
        if which_challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.mask_func = mask_func
        self.resolution = resolution
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(self, kspace, mask, target, attrs, fname, slice):
        """
        Args:
            kspace (numpy.array): Input k-space of shape (num_coils, rows, cols, 2) for multi-coil
                data or (rows, cols, 2) for single coil data.
            mask (numpy.array): Mask from the test dataset
            target (numpy.array): Target image
            attrs (dict): Acquisition related information stored in the HDF5 object.
            fname (str): File name
            slice (int): Serial number of the slice.
        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Zero-filled input image.
                target (torch.Tensor): Target image converted to a torch Tensor.
                mean (float): Mean value used for normalization.
                std (float): Standard deviation value used for normalization.
        """
        kspace = T.to_tensor(kspace)
        # Apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = T.apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace
            mask = torch.from_numpy(mask[:, np.newaxis].astype(np.float32))

        mask = mask.repeat(kspace.shape[0],1,1) # 640 is fixed for single coil 

        # Inverse Fourier Transform to get zero filled solution
        image = T.ifft2(masked_kspace)
        
        # Complex abs
        image_abs = T.complex_abs(image)
        image_abs_max = image_abs.max()
        
        # Image and kspace normalization
        image_norm = image / image_abs_max
        masked_kspace_norm = masked_kspace / image_abs_max
                
        # Crop input image to given resolution if larger
        smallest_width = min(self.resolution, image.shape[-2])
        smallest_height = min(self.resolution, image.shape[-3])
        
        if target is not None:
            smallest_width = min(smallest_width, target.shape[-1])
            smallest_height = min(smallest_height, target.shape[-2])

        crop_size = (smallest_height, smallest_width)
        crop_size_tensor = torch.Tensor(crop_size)
        
        image_norm_crop = T.complex_center_crop(image_norm, crop_size)
        kspace_norm_crop = T.fftshift(T.fft2(image_norm_crop))
        
        if target is not None:
            target = T.to_tensor(target)
            target_norm = target / image_abs_max
        else:
            target_norm = torch.Tensor([0])
            
        return image_norm_crop, kspace_norm_crop, image_norm, masked_kspace_norm, target_norm, mask, image_abs_max, crop_size_tensor, fname, slice


def save_reconstructions(reconstructions, out_dir):
    """
    Saves the reconstructions from a model into h5 files that is appropriate for submission
    to the leaderboard.
    Args:
        reconstructions (dict[str, np.array]): A dictionary mapping input filenames to
            corresponding reconstructions (of shape num_slices x height x width).
        out_dir (pathlib.Path): Path to the output directory where the reconstructions
            should be saved.
    """
    out_dir.mkdir(exist_ok=True)
    for fname, recons in reconstructions.items():
        with h5py.File(out_dir / fname, 'w') as f:
            f.create_dataset('reconstruction', data=recons)

def create_data_loaders(args):


    transform = DataTransform(args.resolution, args.challenge)

    data = SliceDataDev(args.data_path,transform=transform)

    data_loader = DataLoader(
        dataset=data,
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=True,)

    return data_loader


def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model = DnCn(args).to(args.device)    
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])
    return model


def run_unet(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(list)
    with torch.no_grad():
        for (iter,data) in enumerate(tqdm(data_loader)):

            image_crop, kspace_crop, image, kspace, target, mask, abs_max, crop_size_tensor, fnames, slices = data
    
            image_crop = image_crop.permute(0, 3, 1, 2).to(args.device)
            kspace_crop = kspace_crop.permute(0, 3, 1, 2).to(args.device)
            kspace = kspace.to(args.device)
            image = image.permute(0, 3, 1, 2).to(args.device)
            target = target.to(args.device)
            mask = mask.to(args.device)
            abs_max = abs_max.to(args.device)
           
            crop_size = (int(crop_size_tensor.numpy()[0,0]), int(crop_size_tensor.numpy()[0,1]))
    
            #print (image_crop.shape, kspace_crop.shape, image.shape, kspace.shape, mask.shape, crop_size_tensor.shape)
            output = model(image_crop, kspace_crop, image, kspace, mask, crop_size)
            output = T.complex_abs(output.permute(0,2,3,1))
 
            recons = output * abs_max

            recons = recons.to('cpu')

            for i in range(recons.shape[0]):
                recons[i] = recons[i] 
                reconstructions[fnames[i]].append((slices[i].numpy(), recons[i].numpy()))

    reconstructions = {
        fname: np.stack([pred for _, pred in sorted(slice_preds)])
             for fname, slice_preds in reconstructions.items()
         }

    return reconstructions


def main(args):
    
    data_loader = create_data_loaders(args)
    model = load_model(args.checkpoint)

    reconstructions = run_unet(args, model, data_loader)
    save_reconstructions(reconstructions, args.out_dir)


def create_arg_parser():

    parser = argparse.ArgumentParser(description="Valid setup for MR recon U-Net")
    parser.add_argument('--checkpoint', type=pathlib.Path, required=True,
                        help='Path to the U-Net model')
    parser.add_argument('--out-dir', type=pathlib.Path, required=True,
                        help='Path to save the reconstructions to')
    parser.add_argument('--batch-size', default=16, type=int, help='Mini-batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Which device to run on')
    parser.add_argument('--data-path',type=str,help='path to validation dataset')
    parser.add_argument('--resolution', default=320, type=int, help='Resolution of images (brain \
                challenge expects resolution of 384, knee resolution expcts resolution of 320')
    parser.add_argument('--challenge', default = 'singlecoil', choices=['singlecoil', 'multicoil'], help='Which challenge')

    return parser

if __name__ == '__main__':
    args = create_arg_parser().parse_args(sys.argv[1:])
    main(args)
