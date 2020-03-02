import pathlib
import sys
from collections import defaultdict
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import SliceDataDev
from models import UnetModel, DataConsistencyLayer,dAUTOMAP,conv_block,ConvFeatureBlock,UnetModelFeature,DnCn
import h5py
from tqdm import tqdm
from torch.nn import functional as F

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
    data = SliceDataDev(args.data_path,args.acceleration_factor,args.dataset_type)
    data_loader = DataLoader(
        dataset=data,
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=True,
    )

    return data_loader

def build_dclayer(args):

    dc_layer = DataConsistencyLayer(args.usmask_path,args.acceleration_factor,args.device)
    
    return dc_layer


def load_model(args,checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    #args = checkpoint['args']
    #model = UnetModel(2, 1, args.num_chans, args.num_pools, args.drop_prob).to(args.device)
    #model = conv_block(n_ch=2,nd=5,n_out=1).to(args.device)
    #model = conv_block(n_ch=3,nd=5,n_out=1).to(args.device)
    #print(model)
    model = ConvFeatureBlock(3).to(args.device)

    #if args.data_parallel:
    #    model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])
    return model


def load_dautomap(args,checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    #args = checkpoint['args']

    patch_size = 150

    model_params = {
      'input_shape': (2, patch_size, patch_size),
      'output_shape': (1, patch_size, patch_size),
      'tfx_params': {
        'nrow': patch_size,
        'ncol': patch_size,
        'nch_in': 2,
        'kernel_size': 1,
        'nl': 'relu',
        'init_fourier': False,
        'init': 'xavier_uniform_',
        'bias': True,
        'share_tfxs': False,
        'learnable': True,
      },
      'depth': 2,
      'nl':'relu'
    }

    model = dAUTOMAP(model_params['input_shape'],model_params['output_shape'],model_params['tfx_params']).to(args.device)

#    if args.data_parallel:
#        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])

    return model

def load_unet(args,checkpoint_file):

    checkpoint = torch.load(checkpoint_file)
    #args = checkpoint['args']
    model = UnetModel(1,1,32,4,0).to(args.device)
    #print(model)

#    if args.data_parallel:
#        model = torch.nn.DataParallel(model)

    model.load_state_dict(checkpoint['model'])

    return model

def load_lem_unet(args,checkpoint_file):

    checkpoint = torch.load(checkpoint_file)
    model = UnetModelFeature(1,1,32,4,0).to(args.device)
    model.load_state_dict(checkpoint['model'])

    return model

def load_base_model(args,checkpoint_file):

    checkpoint = torch.load(checkpoint_file)
    model = DnCn(args,n_channels=1).to(args.device)
    model.load_state_dict(checkpoint['model'])

    return model 


def run_unet(args, model, dautomap_model,unet_model,lem_model,dncn_model,dc_layer,data_loader):

    model.eval()
    reconstructions = defaultdict(list)
    with torch.no_grad():
        for (iter,data) in enumerate(tqdm(data_loader)):

            input, input_kspace, target,fnames,slices = data

            input = input.float()
          
            #input_kspace = input_kspace.float()
            target = target.float()
            target = target.unsqueeze(1).to(args.device)

            input = input.unsqueeze(1).to(args.device)
            input_crop = input[:,:,5:155,5:155]

            #print("input_kspace size: ",input_kspace.size())
            input_kspace1 = input_kspace.permute(0,3,1,2).to(args.device)
            input_kspace2 = input_kspace.unsqueeze(1).to(args.device)

            dautomap_pred = dautomap_model(input_kspace1.float())
            dautomap_pred = F.pad(dautomap_pred,(5,5,5,5),"constant",0)
            unet_pred  = unet_model(input)

            dncn_pred = dncn_model(input_crop,input_kspace2)
            dncn_pred = F.pad(dncn_pred,(5,5,5,5),"constant",0)
            feat,lem  = lem_model(dncn_pred)

            pred_cat = torch.cat([unet_pred,dautomap_pred,input],dim=1)

            #recons = model(pred_cat,feat)
            recons = model(pred_cat,feat)

            if not dc_layer is None:
                recons = dc_layer(input_kspace,recons)

            recons = recons.to('cpu').squeeze(1)

            if args.dataset_type == 'cardiac':
                recons = recons[:,5:155,5:155]
            
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
    model = load_model(args,args.checkpoint)

    dautomap_model = load_dautomap(args,args.dautomap_model_path)
    unet_model     = load_unet(args,args.unet_model_path)
    lem_model      = load_lem_unet(args,args.seg_unet_path)
    dncn_model     = load_base_model(args,args.dncn_model_path)

    if args.data_consistency:
        dc_layer = build_dclayer(args)
    else:
        dc_layer = None

    reconstructions = run_unet(args, model, dautomap_model, unet_model, lem_model,dncn_model,dc_layer, data_loader)
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
    parser.add_argument('--data_consistency',action='store_true')
    parser.add_argument('--unet_model_path',type=str,help='unet best model path')
    parser.add_argument('--dautomap_model_path',type=str,help='dautomap best model path')
    parser.add_argument('--seg_unet_path',type=str,help='unet model path')
    parser.add_argument('--dncn_model_path',type=str,help='unet model path')
    
    return parser

if __name__ == '__main__':
    args = create_arg_parser().parse_args(sys.argv[1:])
    main(args)
