import pathlib
import sys
from collections import defaultdict
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import SliceData
from architecture import VariationalNetworkModel
import h5py
from tqdm import tqdm
import transforms as T

class DataTransform:
    """
    Data Transformer for training Var Net models.
    """

    def __init__(self, resolution, mask_func=None, use_seed=True):
        """
        Args:
            mask_func (common.subsample.MaskFunc): A function that can create a mask of
                appropriate shape.
            resolution (int): Resolution of the image.
            use_seed (bool): If true, this class computes a pseudo random number generator seed
                from the filename. This ensures that the same mask is used for all the slices of
                a given volume every time.
        """
        self.mask_func = mask_func
        self.resolution = resolution
        self.use_seed = use_seed

    def __call__(self, kspace, mask, target, attrs, fname):
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
                masked_kspace (torch.Tensor): Masked k-space
                mask (torch.Tensor): Mask
                target (torch.Tensor): Target image converted to a torch Tensor.
                fname (str): File name
                slice (int): Serial number of the slice.
                max_value (numpy.array): Maximum value in the image volume
        """
        if target is not None:
            target = T.to_tensor(target)
            max_value = attrs['max']
        else:
            target = torch.tensor(0)
            max_value = 0.0
        kspace = T.to_tensor(kspace)
        seed = None if not self.use_seed else tuple(map(ord, fname))
        acq_start = attrs['padding_left']
        acq_end = attrs['padding_right']
        if self.mask_func:
            masked_kspace, mask = T.apply_mask(kspace, self.mask_func, seed, (acq_start, acq_end))
        else:
            masked_kspace = kspace
            shape = np.array(kspace.shape)
            num_cols = shape[-2]
            shape[:-3] = 1
            mask_shape = [1 for _ in shape]
            mask_shape[-2] = num_cols
            mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))
            mask[:,:,:acq_start] = 0
            mask[:,:,acq_end:] = 0

        return masked_kspace, mask.byte(), target, fname, max_value

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

    transform = DataTransform(args.resolution)

    data = SliceData(
        root=args.data_path,
        root_csv=args.data_csv_path,
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

    model = VariationalNetworkModel(args).to(args.device)
    model.load_state_dict(checkpoint['model'])

    return model


def run_unet(args, model, data_loader):

    model.eval()
    reconstructions = defaultdict(list)
    with torch.no_grad():
        for (iter,data) in enumerate(tqdm(data_loader)):

            masked_kspace, mask, _, fnames, _ = data

            masked_kspace = masked_kspace.to(args.device)
            mask = mask.to(args.device)

            output = model(masked_kspace, mask)
            recons = T.center_crop(output, (args.resolution, args.resolution))
            recons = recons.cpu().numpy()

            for i in range(recons.shape[0]):
                fname, slice = fnames[i].split('/')[-2], int(fnames[i].split('/')[-1][:-3]), 
                #print (fname, slice)
                reconstructions[fname].append((slice, recons[i]))

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
    parser.add_argument('--data-path',type=pathlib.Path,help='path to validation dataset')
    parser.add_argument('--data-csv-path',type=pathlib.Path,help='path to validation dataset csv')
    parser.add_argument('--resolution', default=320, type=int, help='Resolution of images (brain \
                challenge expects resolution of 384, knee resolution expcts resolution of 320')
    parser.add_argument('--challenge', default = 'singlecoil', choices=['singlecoil', 'multicoil'], help='Which challenge')

    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args(sys.argv[1:])
    main(args)
