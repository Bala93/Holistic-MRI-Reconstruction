import pathlib
import sys
from collections import defaultdict
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import SliceDataDev
from architecture import network
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

    def __call__(self, kspace, sensitivity, mask, fname, slice):
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
        sensitivity = T.to_tensor(sensitivity)

        # Apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = T.apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace
            mask = torch.from_numpy(mask[None,None,:,None].astype(np.float32))

        ## Check the dimension and change accordingly
        mask = mask.repeat(kspace.shape[0],kspace.shape[1],1,2) 

        ## normalization factor 
        masked_image = T.ifft2(masked_kspace)
        masked_image_rss = T.root_sum_of_squares(T.complex_abs(masked_image),dim=0)
        masked_image_rss_max = masked_image_rss.max()

        ## kspace normalization 
        masked_kspace_norm = masked_kspace / masked_image_rss_max

        # Inverse Fourier Transform to get zero filled solution
        masked_image_norm = T.ifft2(masked_kspace_norm)
        
        ## combine channel and sensitivity map
        masked_image_norm_reduce = T.complex_mul(masked_image_norm, T.complex_conj(sensitivity)).sum(dim=0) ## Fu network input 
        masked_image_norm_reduce_crop = T.complex_center_crop(masked_image_norm_reduce, (320,320)) ## II networt input
        masked_kspace_norm_reduce_crop = T.fftshift(T.fft2(masked_image_norm_reduce_crop)) ## KI network input TODO: Check fftshift 


        fname = fname + '.h5'
        slice = int(slice[:-3])

        return masked_image_norm_reduce_crop, masked_kspace_norm_reduce_crop, masked_image_norm_reduce, sensitivity, mask, masked_kspace_norm, masked_image_rss_max, fname, slice




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

    data = SliceDataDev(
        root=args.data_path,
        csv_path=args.data_csv_path,
        transform=transform,
        challenge=args.challenge)

    loader = DataLoader(
        dataset=data,
        batch_size=args.batch_size,
        #shuffle=True,
        #num_workers=64,
        #pin_memory=True,
    )

    return loader


def load_model(checkpoint_file):

    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']

    dccoeff = 0.1
    wacoeff = 0.1
    cascade = 3

    model = network(dccoeff,wacoeff,cascade).to(args.device)
    model.load_state_dict(checkpoint['model'])

    return model


def run_unet(args, model, data_loader):

    model.eval()
    reconstructions = defaultdict(list)
    with torch.no_grad():
        for (iter,data) in enumerate(tqdm(data_loader)):

            img_und_crop, img_und_kspace, img_und, sensitivity, masks, rawdata_und, rss_max, fnames, slices = data
    
            rawdata_und = rawdata_und.to(args.device)
            masks = masks.to(args.device)
    
            img_und_crop = img_und_crop.to(args.device)
            img_und = img_und.to(args.device)
            img_und_kspace = img_und_kspace.to(args.device)
            sensitivity = sensitivity.to(args.device)

            rss_max = rss_max.to(args.device)
            
            output = model(img_und_crop, img_und_kspace, img_und, rawdata_und,masks,sensitivity,(320,320))
           
            output_expand = T.complex_mul(output, sensitivity)
            recons = T.root_sum_of_squares(T.complex_abs(T.complex_center_crop(output_expand,(320,320))),dim=1)
            recons = recons * rss_max
            recons = recons.cpu()

            for i in range(recons.shape[0]):
                recons[i] = recons[i] 
                reconstructions[fnames[i]].append((slices[i].numpy(), recons[i].numpy()))

            #break 

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
    parser.add_argument('--data-csv-path',type=str,help='path to validation dataset csv')
    parser.add_argument('--resolution', default=320, type=int, help='Resolution of images (brain \
                challenge expects resolution of 384, knee resolution expcts resolution of 320')
    parser.add_argument('--challenge', default = 'singlecoil', choices=['singlecoil', 'multicoil'], help='Which challenge')

    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args(sys.argv[1:])
    main(args)
