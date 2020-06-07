import pathlib
import sys
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils import save_reconstructions
#from mri_data import SliceData, SliceDataDev
from dataset import KneeDataDev
from models import UNet
from architecture import networkRefine
from tqdm import tqdm
import argparse
from data import transforms as T

def create_data_loaders(args):
    data = KneeDataDev(args.val_path)

    data_loader = DataLoader(
        dataset=data,
        batch_size=args.batch_size,
        #num_workers=32,
        pin_memory=True,
    )

    return data_loader

def build_model(args):

    model = networkRefine(args, mode='valid').to(args.device)

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

            img_gt,img_und,img_und_kspace,rawdata_und,masks,sensitivity,fnames = data

            target = img_gt.to(args.device)
            img_und = img_und.to(args.device)
            img_und_kspace = img_und_kspace.to(args.device)
            rawdata_und = rawdata_und.to(args.device)
            masks = masks.to(args.device)
            sensitivity = sensitivity.to(args.device)

            output = model(img_und,img_und_kspace,rawdata_und,masks,sensitivity)

            recons = T.complex_abs(output).to('cpu')

            #print (recons.shape)
 
            for i in range(recons.shape[0]):
                reconstructions[fnames[i]].append(recons[i].numpy())

    reconstructions = {
        fname: np.stack([pred for pred in sorted(slice_preds)])
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
    parser.add_argument('--acceleration', type=int, choices=[2, 4, 8], default=4, help='Acceleration factor for undersampled data')
    parser.add_argument('--checkpoint', type=pathlib.Path, required=True,
                        help='Path to the U-Net model')
    parser.add_argument('--out-dir', type=pathlib.Path, required=True,
                        help='Path to save the reconstructions to') 
    parser.add_argument('--batch-size', default=1, type=int, help='Mini-batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Which device to run on')
  
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args(sys.argv[1:])
    main(args)
