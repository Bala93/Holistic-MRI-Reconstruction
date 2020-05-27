import pathlib
import sys
from collections import defaultdict
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset_1 import SliceDataDev
from torch.nn import functional as F
from models import DnCn
import h5py
from tqdm import tqdm
from collections import OrderedDict
import os

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
    if (not (os.path.exists(out_dir))):
        os.mkdir(out_dir)
    out_dir.mkdir(exist_ok=True)
    print('Saved directory is',out_dir)
    for fname, recons in reconstructions.items():
        with h5py.File(out_dir / fname, 'w') as f:
            f.create_dataset('reconstruction', data=recons)


def create_data_loaders(args):

    data = SliceDataDev(args.data_path,args.acceleration_factor,args.dataset_type)
    data_loader = DataLoader(
        dataset=data,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
    )

    return data_loader

def load_model(args,checkpoint_dccnn,checkpoint_percept,alpha):
 
    checkpoint_dccnn = torch.load(checkpoint_dccnn)
    args_dccnn = checkpoint_dccnn['args']
    checkpoint_percept = torch.load(checkpoint_percept)
    args_percept = checkpoint_percept['args']
    
    model_dccnn = DnCn(args,n_channels=1).to(args.device)
    model_percept = DnCn(args_percept,n_channels=1).to(args.device)
    model_combined = DnCn(args,n_channels=1).to(args.device)
    
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

            input,input_kspace,target,fnames,slices = data

            input_kspace = input_kspace.float()
            input_kspace = input_kspace.unsqueeze(1).to(args.device).double()
            input = input.unsqueeze(1).to(args.device)

            input = input.float()


            recons = models_combined(input,input_kspace) 

            #recons = recons + input 

            recons = recons.to('cpu').squeeze(1)
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
    print(args.out_dir)
    if not(os.path.exists(args.out_dir)):
        os.mkdir(args.out_dir)
    data_loader = create_data_loaders(args)
    all_alphas = list(np.linspace(0,1,11))
    for alpha in all_alphas:
        alpha = round(alpha,2);
        model_dc = load_model(args,args.checkpoint_dccnn,args.checkpoint_percept,alpha)
        reconstructions = run_gan(args, model_dc,data_loader)
        out_dir = args.out_dir / str(alpha)
        save_reconstructions(reconstructions, out_dir)
   
def create_arg_parser():

    parser = argparse.ArgumentParser(description="Valid setup for MR recon U-Net")
    parser.add_argument('--checkpoint_dccnn', type=pathlib.Path, required=True,
                        help='Path to the dc_cnn model')
    parser.add_argument('--num-pools', type=int, default=4, help='Number of U-Net pooling layers')
    parser.add_argument('--drop-prob', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--num-chans', type=int, default=32, help='Number of U-Net channels')
    parser.add_argument('--checkpoint_percept', type=pathlib.Path, required=True,help='Path to the dc_cnn model')
    parser.add_argument('--out-dir', type=pathlib.Path, required=True,
                        help='Path to save the reconstructions to')
    parser.add_argument('--batch-size', default=16, type=int, help='Mini-batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Which device to run on')
    parser.add_argument('--data-path',type=str,help='path to validation dataset')

    parser.add_argument('--acceleration_factor',type=str,help='acceleration factors')

    parser.add_argument('--usmask_path',type=str,help='Path to USMASK')
    parser.add_argument('--dataset_type',type=str,help='cardiac,kirby')
    
    return parser

if __name__ == '__main__':
    args = create_arg_parser().parse_args(sys.argv[1:])
    main(args)
