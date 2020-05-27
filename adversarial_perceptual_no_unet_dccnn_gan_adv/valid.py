import pathlib
import sys
from collections import defaultdict
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import SliceData_mod_dev
from torch.nn import functional as F
from models import UnetModel,DnCn
import h5py
from tqdm import tqdm

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

    data = SliceData_mod_dev(args.data_path)
    data_loader = DataLoader(
        dataset=data,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
    )

    return data_loader

def load_model(checkpoint_file):

    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model_dc = DnCn(args,n_channels=1).to(args.device)

    #print(model)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)

    model_dc.load_state_dict(checkpoint['modelG_dc'])

    return model_dc


def run_gan(args, model_dc,data_loader):

    model_dc.eval()

    reconstructions = defaultdict(list)
    with torch.no_grad():
        for (iter,data) in enumerate(tqdm(data_loader)):

            input,input_kspace,target,fnames = data

            input = input.unsqueeze(1).to(args.device)
            input_kspace = input_kspace.unsqueeze(1).to(args.device)
            input = input.float()

            recons = model_dc(input,input_kspace) 

            #recons = recons + input 

            recons = recons.to('cpu').squeeze(1)

            reconstructions[fnames[0]].append(recons.numpy())

    # reconstructions = {
    #     fname: np.stack([pred for _, pred in sorted(slice_preds)])
    #     for fname, slice_preds in reconstructions.items()
    # }

    return reconstructions


def main(args):
    print(args.out_dir)
    data_loader = create_data_loaders(args)
    model_dc = load_model(args.checkpoint)
    reconstructions = run_gan(args, model_dc,data_loader)
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

    parser.add_argument('--acceleration_factor',type=str,help='acceleration factors')

    parser.add_argument('--usmask_path',type=str,help='Path to usmask')
    parser.add_argument('--dataset_type',type=str,help='cardiac,kirby')
    
    return parser

if __name__ == '__main__':
    args = create_arg_parser().parse_args(sys.argv[1:])
    main(args)
