import pathlib
import sys
from collections import defaultdict
import argparse
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from dataset import SliceDataEvaluateDev
from models import DnCn,UnetModel,DnCnFeature,DnCnFeatureLoop,UnetModelTakeLatentDecoder,UnetModelTakeEverywhereWithIntermediate,DnCnFeatureLoopAssistOnlyFirstBlock
import h5py
from tqdm import tqdm
import torch.nn as nn
import os
from chattn import CSEUnetModelTakeLatentDecoder

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

    #data = SliceDataDev(args.data_path,args.acceleration_factor,args.dataset_type,args.usmask_path)
    data = SliceDataEvaluateDev(args.data_path,args.acceleration_factor,args.dataset_type,args.dncn_model_path)
    data_loader = DataLoader(
        dataset=data,
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=True,
    )

    return data_loader

def load_segmodel(args):
    
    #model = UnetModel(
    #     in_chans=1,
    #     out_chans=1,
    #     chans=32,
    #     num_pool_layers=4,
    #     drop_prob=0.0
    # ).to(args.device)
    model = UnetModelTakeLatentDecoder(
    #model = UnetModelTakeEverywhereWithIntermediate(
         in_chans=1,
         out_chans=1,
         chans=32,
         num_pool_layers=4,
         drop_prob=0.0
     ).to(args.device)
#    model = CSEUnetModelTakeLatentDecoder(
#        in_chans=1,
#        out_chans=1,
#        chans=32,
#        num_pool_layers=4,
#        drop_prob=0.0,
#        attention_type='cSE',
#        reduction=16
#    ).to(args.device)

   
    checkpoint = torch.load(args.unet_model_path)
    model.load_state_dict(checkpoint['model'])
    model = nn.DataParallel(model).to(args.device)
    return model

def load_recmodel(args):

    checkpoint = torch.load(args.dncn_model_path)
    args.usmask_path = checkpoint['args'].usmask_path # to add mask path to args 
    model = DnCn(args,n_channels=1).to(args.device)
    model.load_state_dict(checkpoint['model'])
    return model


def load_model(checkpoint_file):

    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
 ##   model = DnCn(args,n_channels=1).to(args.device)
    #print(model)
    #model = UnetModel(1, 1, args.num_chans, args.num_pools, args.drop_prob).to(args.device)
    #model = DnCnFeature(args,n_channels=1).to(args.device)
    model = DnCnFeatureLoop(args,n_channels=1).to(args.device)
    #model = DnCnFeatureLoopAssistOnlyFirstBlock(args,n_channels=1).to(args.device)
    #model = torch.nn.DataParallel(model).to(args.device)
    #if args.data_parallel:
    model.load_state_dict(checkpoint['model'])
    return model


#def run_unet(args, segmodel, recmodel, model, data_loader):
def run_unet(args, segmodel, model, data_loader):

    model.eval()
    #recmodel.eval()
    segmodel.eval()
    
    us_mask_path = os.path.join(args.usmask_path,'mask_{}.npy'.format(args.acceleration_factor))
    us_mask = torch.from_numpy(np.load(us_mask_path)).float()
    us_mask = us_mask.to(args.device)
    reconstructions = defaultdict(list)
    with torch.no_grad():
        for (iter,data) in enumerate(tqdm(data_loader)):

            input, input_kspace, target,predictedimg, fnames,slices,t1imgfs = data
            input = input.unsqueeze(1).to(args.device)
            t1imgfs = t1imgfs.unsqueeze(1).float().to(args.device)
            input_kspace = input_kspace.permute(0,3,1,2).to(args.device)
            #input_kspace = input_kspace.unsqueeze(1).to(args.device)

            us_mask1 = us_mask.repeat(input.shape[0],1,1).unsqueeze(1) # 256,256 to 4, 256,256  to 4,1,256,256 after unsqueeze
            input = input.float()
            input_kspace = input_kspace.float()
            rec = predictedimg
            rec = rec.unsqueeze(1).to(args.device)

            #rec = recmodel(input,input_kspace)

            if args.dataset_type == 'cardiac':
                rec = F.pad(rec,(5,5,5,5),"constant",0)

            feat,seg = segmodel(rec)

            if args.dataset_type == 'cardiac':
                feat = feat[:,:,5:155,5:155]

            recons = model(input, input_kspace,feat,us_mask1,seg,t1imgfs).to('cpu').squeeze(1)

            #if args.dataset_type == 'cardiac':
            #    recons = recons[:,5:155,5:155]

            
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
    segmodel = load_segmodel(args)
    #recmodel = load_recmodel(args)
    reconstructions = run_unet(args, segmodel,model, data_loader)
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
    parser.add_argument('--dataset_type',type=str,help='cardiac,kirby')
    parser.add_argument('--usmask_path',type=str,help='undersampling mask path')
    parser.add_argument('--unet_model_path',type=str,help='unet model path')
    parser.add_argument('--dncn_model_path',type=str,help='dncn model path')
    
    return parser

if __name__ == '__main__':
    args = create_arg_parser().parse_args(sys.argv[1:])
    main(args)
