import pathlib
import sys
from collections import defaultdict
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import SliceDataDev
from torch.nn import functional as F
from models import UnetModel,DnCn
import h5py
from tqdm import tqdm
from collections import OrderedDict
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

    data = SliceDataDev(args.train_path,args.acceleration_factor,args.dataset_type)
    data_loader = DataLoader(
        dataset=data,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
    )

    return data_loader

def load_model(checkpoint_dccnn,checkpoint_percept,alpha):

    checkpoint_dccnn = torch.load(checkpoint_dccnn)
    args_dccnn = checkpoint_dccnn['args']
    checkpoint_percept = torch.load(checkpoint_percept)
    args_percept = checkpoint_percept['args']
    
    model_dccnn = DnCn(args_dccnn,n_channels=1).to(args.device)
    model_percept = DnCn(args_percept,n_channels=1).to(args.device)
    model_combined = DnCn(args_dccnn,n_channels=1).to(args.device)
    
    if args_dccnn.data_parallel:
        model_dccnn = torch.nn.DataParallel(model_dccnn)
    if args_percept.data_parallel:
        model_percept = torch.nn.DataParallel(model_percept)

    model_dccnn.load_state_dict(checkpoint_dccnn['model'])
    model_percept.load_state_dict(checkpoint_percept['modelG_dc'])
   
    param_combined = OrderedDict() 
    for param_name in checkpoint_dccnn['model']:
        param_dccnn = checkpoint_dccnn['model'][param_name]
        param_percept = checkpoint_percept['modelG_dc'][param_name]
        param_combined[param_name] = (1 - alpha) * param_dccnn + alpha * param_percept 
    

    model_combined.load_state_dict(param_combined)
    
    return model_combined


def run_gan(args, models_combined,data_loader):

    models_combined.eval()

    reconstructions = defaultdict(list)
    with torch.no_grad():
        for (iter,data) in enumerate(tqdm(data_loader)):

            input,input_kspace,target,fnames = data

            input = input.unsqueeze(1).to(args.device)
            input_kspace = input_kspace.permute(0,3,1,2).to(args.device)
            input = input.float()

            recons = models_combined(input,input_kspace) 

            #recons = recons + input 

            recons = recons.to('cpu').squeeze(1)

            reconstructions[fnames[0]].append(recons.numpy())

     reconstructions = {
         fname: np.stack([pred for _, pred in sorted(slice_preds)])
         for fname, slice_preds in reconstructions.items()
     }

    return reconstructions


def main(args):
    print(args.out_dir)
    data_loader = create_data_loaders(args)
    model_dc = load_model(args.checkpoint_dccnn,args.checkpoint_percept,args.interp_factor)
    reconstructions = run_gan(args, model_dc,data_loader)
    save_reconstructions(reconstructions, args.out_dir)


def create_arg_parser():

    parser = argparse.ArgumentParser(description="Valid setup for MR recon U-Net")
    parser.add_argument('--checkpoint_dccnn', type=pathlib.Path, required=True,
                        help='Path to the dc_cnn model')
    parser.add_argument('--checkpoint_percept', type=pathlib.Path, required=True,help='Path to the dc_cnn model')
    parser.add_argument('--out-dir', type=pathlib.Path, required=True,
                        help='Path to save the reconstructions to')
    parser.add_argument('--batch-size', default=16, type=int, help='Mini-batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Which device to run on')
    parser.add_argument('--data-path',type=str,help='path to validation dataset')

    parser.add_argument('--acceleration_factor',type=str,help='acceleration factors')

    parser.add_argument('--dataset_type',type=str,help='cardiac,kirby')
    parser.add_argument('--interp_factor',type=float,help='interpolation factor (0-1)')
    
    return parser

if __name__ == '__main__':
    args = create_arg_parser().parse_args(sys.argv[1:])
    main(args)
