import pathlib
import sys
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils import save_reconstructions
#from mri_data import SliceData, SliceDataDev
from dataset import SliceDataEvaluateDev
from models import UnetModelTakeLatentDecoder
from architecture import DnCnRefine
from tqdm import tqdm
import argparse
import os 


def create_data_loaders(args):

    data = SliceDataEvaluateDev(args.val_path,args.acceleration,args.dataset_type,args.dncn_model_path)

    data_loader = DataLoader(
        dataset=data,
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=True,
    )

    return data_loader


def build_segmodel(args):
    
    model = UnetModelTakeLatentDecoder(
        in_chans=1,
        out_chans=1,
        chans=32,
        num_pool_layers=4,
        drop_prob=0,
    ).to(args.device)

    checkpoint = torch.load(args.seg_unet_path)
    model.load_state_dict(checkpoint['model'])

    for params in model.parameters():
        params.requires_grad=False 

    return model


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

def run(args, model, unetmodel, data_loader, us_mask):
    model.eval()
    reconstructions = defaultdict(list)
    with torch.no_grad():
        for (input, kspace, target, predictedimg, fnames, slices, t1imgfs) in tqdm(data_loader):

            input = input.unsqueeze(1).to(args.device)
            kspace = kspace.to(args.device)
            kspace = kspace.permute(0,3,1,2)
            t1imgfs = t1imgfs.unsqueeze(1).to(args.device)

            input = input.float()
            kspace = kspace.float()
            t1imgfs = t1imgfs.float()

            rec = predictedimg
            rec = rec.unsqueeze(1).to(args.device)


            feat,seg = unetmodel(rec)
            us_mask1 = us_mask.repeat(input.shape[0],1,1).unsqueeze(1) # 256,256 to 4, 256,256  to 4,1,256,256 after unsqueeze

            recons = model(input, kspace, feat, us_mask1, seg, t1imgfs).to('cpu').squeeze(1)
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
    segmodel = build_segmodel(args)

    us_mask_path = os.path.join(args.usmask_path,'mask_{}x.npy'.format(args.acceleration))
    us_mask = torch.from_numpy(np.load(us_mask_path)).float().to(args.device)

    print('running model...')
    reconstructions = run(args, model, segmodel, data_loader, us_mask)
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
    parser.add_argument('--dataset_type',type=str,help='cardiac,kirby') 
    parser.add_argument('--usmask_path',type=str,help='us mask path')
    parser.add_argument('--seg_unet_path',type=str,help='unet segmentation path')
    parser.add_argument('--dncn_model_path',type=str,help='dncn model path')
  
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args(sys.argv[1:])
    main(args)
