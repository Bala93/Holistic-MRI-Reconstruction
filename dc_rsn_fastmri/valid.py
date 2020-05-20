import pathlib
import sys
from collections import defaultdict
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import SliceDataDev
from models import UnetModel
import h5py
from tqdm import tqdm
import transforms


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
        kspace = transforms.to_tensor(kspace)
        # Apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = transforms.apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace

        # Inverse Fourier Transform to get zero filled solution
        image = transforms.ifft2(masked_kspace)
        # Crop input image to given resolution if larger
        smallest_width = min(self.resolution, image.shape[-2])
        smallest_height = min(self.resolution, image.shape[-3])
        if target is not None:
            smallest_width = min(smallest_width, target.shape[-1])
            smallest_height = min(smallest_height, target.shape[-2])

        crop_size = (smallest_height, smallest_width)
        image = transforms.complex_center_crop(image, crop_size)
        # Absolute value
        image = transforms.complex_abs(image)
        # Apply Root-Sum-of-Squares if multicoil data
        if self.which_challenge == 'multicoil':
            image = transforms.root_sum_of_squares(image)
        # Normalize input
        image, mean, std = transforms.normalize_instance(image, eps=1e-11)
        image = image.clamp(-6, 6)
        # Normalize target
        if target is not None:
            target = transforms.to_tensor(target)
            target = transforms.center_crop(target, crop_size)
            target = transforms.normalize(target, mean, std, eps=1e-11)
            target = target.clamp(-6, 6)
        else:
            target = torch.Tensor([0])
        return image, target, mean, std, fname, slice





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
    model = UnetModel(1, 1, args.num_chans, args.num_pools, args.drop_prob).to(args.device)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])
    return model


def run_unet(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(list)
    with torch.no_grad():
        for (iter,data) in enumerate(tqdm(data_loader)):

            input, _, mean, std, fnames, slices = data
             
            input = input.unsqueeze(1).to(args.device)
            mean = mean.unsqueeze(1).unsqueeze(2).unsqueeze(3).to(args.device)
            std = std.unsqueeze(1).unsqueeze(2).unsqueeze(3).to(args.device)

            recons = model(input)

            recons = recons * std + mean

            recons = recons.to('cpu').squeeze(1)

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
