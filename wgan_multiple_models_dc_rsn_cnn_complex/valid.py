import pathlib
import sys
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils import save_reconstructions
#from mri_data import SliceData, SliceDataDev
from dataset import SliceData, SliceDataDev
from models import UNet
from architecture import DnCnRefine
from tqdm import tqdm
import argparse
from utils import complex_abs

def create_data_loaders(args):
    data = SliceDataDev(args.val_path, args.acceleration, args.dataset_type)
    data_loader = DataLoader(
        dataset=data,
        batch_size=args.batch_size,
        #num_workers=32,
        pin_memory=True,
    )
    return data_loader

def build_model(args):
    model = DnCnRefine(args,nc=3,mode='valid').to(args.device)
    return model

def load_model(args,checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    #args = checkpoint['args']
    model = build_model(args)
    #if args.data_parallel:
    #model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])

    return checkpoint, model

def run(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(list)
    with torch.no_grad():
        for data in tqdm(data_loader):

            input, kspace, target, fnames, slices = data

            input = input.to(args.device)
            kspace = kspace.to(args.device)
            target = target.to(args.device)

            input = input.float()
            target = target.float()
            kspace = kspace.float()
         
            output = model(input,kspace)

            recons = complex_abs(output.permute(0,2,3,1)).to('cpu')

            #recons = recons[:, 5:155, 5:155]
            for i in range(recons.shape[0]):
                reconstructions[fnames[i]].append((slices[i].numpy(), recons[i].numpy()))

    reconstructions = {
        fname: np.stack([pred for _, pred in sorted(slice_preds)])
        for fname, slice_preds in reconstructions.items()
    }
    return reconstructions


def main(args):
    print('creating data loaders...')
    data_loader = create_data_loaders(args)
    print('loading model...')
    checkpoint, model = load_model(args,args.checkpoint)
    print('running model...')
    reconstructions = run(args, model, data_loader)
    save_reconstructions(reconstructions, args.out_dir)


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val-path', type=str, required=True, help='Path to validation data')
    parser.add_argument('--acceleration', type=int,help='Acceleration factor for undersampled data')
    parser.add_argument('--checkpoint', type=pathlib.Path, required=True,
                        help='Path to the U-Net model')
    parser.add_argument('--out-dir', type=pathlib.Path, required=True,
                        help='Path to save the reconstructions to') 
    parser.add_argument('--batch-size', default=1, type=int, help='Mini-batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Which device to run on')
    parser.add_argument('--dataset_type',type=str,help='cardiac,kirby') 
    parser.add_argument('--usmask_path',type=str,help='us mask path')
  
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args(sys.argv[1:])
    main(args)
