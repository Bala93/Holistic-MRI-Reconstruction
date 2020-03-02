import sys
import logging
import pathlib
import random
import shutil
import time
import functools
import numpy as np
import argparse

import torch
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader
from dataset import SliceData
from models import UnetModel,dAUTOMAP,conv_block,ConvFeatureBlock,UnetModelFeature,DnCn
import torchvision
from torch import nn
from torch.autograd import Variable
from torch import optim
from tqdm import tqdm
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_datasets(args):

    
    #train_data = SliceData(args.train_path,args.acceleration_factor,args.dataset_type,args.usmask_path)
    #dev_data = SliceData(args.validation_path,args.acceleration_factor,args.dataset_type,args.usmask_path)

    train_data = SliceData(args.train_path,args.acceleration_factor,args.dataset_type)
    dev_data = SliceData(args.validation_path,args.acceleration_factor,args.dataset_type)

    return dev_data, train_data


def create_data_loaders(args):
    dev_data, train_data = create_datasets(args)

    display_data = [dev_data[i] for i in range(0, len(dev_data), len(dev_data) // 16)]

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        #num_workers=64,
        #pin_memory=True,
    )
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        #num_workers=64,
        #pin_memory=True,
    )
    display_loader = DataLoader(
        dataset=display_data,
        batch_size=16,
        #num_workers=64,
        #pin_memory=True,
    )
    return train_loader, dev_loader, display_loader


def train_epoch(args, epoch, model_re,model_conv1x1,dautomap_model,unet_model,lem_model,dncn_model,data_loader, optimizer, writer):
    
    model_re.train()
    model_conv1x1.train()

    avg_loss = 0.
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)
    #print ("Entering Train epoch")

    for iter, data in enumerate(tqdm(data_loader)):

        #print (data)

        #print ("Received data from loader")
        input,input_kspace,target = data # Return kspace also we can ignore that for train and test 

        input = input.float()
        target = target.float()
        input = input.unsqueeze(1).to(args.device)
        input_crop = input[:,:,5:155,5:155]
        target = target.unsqueeze(1).to(args.device)

        input_kspace1 = input_kspace.permute(0,3,1,2).to(args.device)
        input_kspace2 = input_kspace.unsqueeze(1).to(args.device)

        dautomap_pred = dautomap_model(input_kspace1.float())
        dautomap_pred = F.pad(dautomap_pred,(5,5,5,5),"constant",0)

        unet_pred     = unet_model(input)

        dncn_pred     = dncn_model(input_crop,input_kspace2)
        dncn_pred     = F.pad(dncn_pred,(5,5,5,5),"constant",0)

        feat,lem      = lem_model(dncn_pred)
        
        feat = model_conv1x1(feat)

        pred_cat = torch.cat([unet_pred,dautomap_pred,input],dim=1)
        output = model_re(pred_cat,feat) # Sending the concatenated result to network 

        loss = F.l1_loss(output,target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()
        writer.add_scalar('TrainLoss',loss.item(),global_step + iter )


        if iter % args.report_interval == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} Avg Loss = {avg_loss:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
        start_iter = time.perf_counter()
        #break

    return avg_loss, time.perf_counter() - start_epoch


def evaluate(args, epoch, model_re, model_conv1x1, dautomap_model,unet_model,lem_model,dncn_model,data_loader, writer):

    model_re.eval()
    model_conv1x1.eval()

    losses = []
    start = time.perf_counter()
    
    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):
    
            input,input_kspace, target = data # Return kspace also we can ignore that for train and test

            input = input.float()
            target = target.float()
            input = input.unsqueeze(1).to(args.device)
            input_crop = input[:,:,5:155,5:155]
            target = target.unsqueeze(1).to(args.device)

            input_kspace1 = input_kspace.permute(0,3,1,2).to(args.device)
            input_kspace2 = input_kspace.unsqueeze(1).to(args.device)

            dautomap_pred = dautomap_model(input_kspace1.float())
            dautomap_pred = F.pad(dautomap_pred,(5,5,5,5),"constant",0) # For cnn this might not be required. 

            unet_pred     = unet_model(input)

            dncn_pred     = dncn_model(input_crop,input_kspace2)
            dncn_pred     = F.pad(dncn_pred,(5,5,5,5),"constant",0)

            feat,lem = lem_model(dncn_pred)

            feat = model_conv1x1(feat)
        
            pred_cat = torch.cat([unet_pred,dautomap_pred,input],dim=1)
            output = model_re(pred_cat,feat) # Sending the concatenated result to network 
    
            #output = model(pred_cat)
            #loss = F.mse_loss(output,target, size_average=False)
            loss = F.mse_loss(output,target)
            
            losses.append(loss.item())
            #break
            
        writer.add_scalar('Dev_Loss',np.mean(losses),epoch)
       
    return np.mean(losses), time.perf_counter() - start


def visualize(args, epoch, model_re, model_conv1x1,dautomap_model, unet_model, lem_model,dncn_model,data_loader, writer):
    
    def save_image(image, tag):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)

    model_re.eval()
    model_conv1x1.eval()

    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):
            input,input_kspace,target = data # Return kspace also we can ignore that for train and test

            input = input.float()
            target = target.float()
            input = input.unsqueeze(1).to(args.device)
            input_crop = input[:,:,5:155,5:155]
            target = target.unsqueeze(1).to(args.device)

            input_kspace1 = input_kspace.permute(0,3,1,2).to(args.device)
            input_kspace2 = input_kspace.unsqueeze(1).to(args.device)
            
            dautomap_pred = dautomap_model(input_kspace1.float())
            dautomap_pred = F.pad(dautomap_pred,(5,5,5,5),"constant",0)

            unet_pred     = unet_model(input)

            dncn_pred     = dncn_model(input_crop,input_kspace2)
            dncn_pred     = F.pad(dncn_pred,(5,5,5,5),"constant",0)
          
            feat,lem = lem_model(dncn_pred)
            feat = model_conv1x1(feat)

            pred_cat = torch.cat([unet_pred,dautomap_pred,input],dim=1)
            output = model_re(pred_cat,feat)

            #print("input: ", torch.min(input), torch.max(input))
            #print("target: ", torch.min(target), torch.max(target))
            #print("predicted: ", torch.min(output), torch.max(output))
            save_image(input, 'Input')
            save_image(dautomap_pred,'dautomap output')
            save_image(unet_pred,'unet output')
            save_image(target, 'Target')
            save_image(output, 'Reconstruction')
            save_image(torch.abs(target.float() - output.float()), 'Error')
            break

def save_model(args, exp_dir, epoch, model_re, model_conv1x1,optimizer,best_dev_loss,is_new_best):

    out = torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model_re': model_re.state_dict(),
            'model_conv1x1': model_conv1x1.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir':exp_dir
        },
        f=exp_dir / 'model.pt'
    )

    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')

'''
def build_model(args):
    
    model = UnetModel(
        in_chans=2,
        out_chans=1,
        chans=args.num_chans,
        num_pool_layers=args.num_pools,
        drop_prob=args.drop_prob
    ).to(args.device)
    
    return model
'''

def build_model(args):
    
    #model = conv_block(n_ch=3,nd=5,n_out=1).to(args.device)
    model_re = ConvFeatureBlock(3).to(args.device)
    model_conv1x1 = nn.Conv2d(1472,32,kernel_size=1).to(args.device)
    
    return model_re,model_conv1x1

def load_dautomap(args,checkpoint_file,is_load=True,is_freeze=True):
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

    #if args.data_parallel:
    #    model = torch.nn.DataParallel(model)

    if is_load:
        model.load_state_dict(checkpoint['model'])

    if is_freeze:
        for param in model.parameters():
            param.requires_grad = False

    return model

def load_unet(args,checkpoint_file,is_load=True,is_freeze=True):

    checkpoint = torch.load(checkpoint_file)
    #args = checkpoint['args']
    model = UnetModel(1, 1, args.num_chans, args.num_pools, args.drop_prob).to(args.device)
    #print(model)
    #if args.data_parallel:
    #    model = torch.nn.DataParallel(model)

    if is_load:
        model.load_state_dict(checkpoint['model'])

    if is_freeze:
        for param in model.parameters():
            param.requires_grad = False

    return model

def load_lem_unet(args,checkpoint_file,is_load=True,is_freeze=True):

    checkpoint = torch.load(checkpoint_file)
    model = UnetModelFeature(1,1,32,4,0).to(args.device)

    if is_load:
        model.load_state_dict(checkpoint['model'])

    if is_freeze:
        for param in model.parameters():
            param.requires_grad = False

    return model 

def load_base_model(args,checkpoint_file,is_load=True,is_freeze=True):

    checkpoint = torch.load(checkpoint_file)
    model = DnCn(args,n_channels=1).to(args.device)
 
    if is_load:
        model.load_state_dict(checkpoint['model'])

    if is_freeze:
        for param in model.parameters():
            param.requires_grad = False

    return model 


def build_optim(args, params):
    optimizer = torch.optim.Adam(params, args.lr, weight_decay=args.weight_decay)
    return optimizer


def main(args):

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    #writer = SummaryWriter(logdir=str(args.exp_dir / 'summary'))
    writer = SummaryWriter(log_dir=str(args.exp_dir / 'summary'))

    model_re,model_conv1x1 = build_model(args)

    optimizer = build_optim(args, list(model_re.parameters()) + list(model_conv1x1.parameters()))
    print ("Model Built")
    print ("Optmizer initialized")
    #if args.data_parallel:
    #    model = torch.nn.DataParallel(model)    

    best_dev_loss = 1e9
    start_epoch = 0

    #logging.info(args)
    #logging.info(model)

    train_loader, dev_loader, display_loader = create_data_loaders(args)
    print ("Dataloader initialized")

    dautomap_model = load_dautomap(args,args.dautomap_model_path,is_load=True,is_freeze=True)
    unet_model     = load_unet(args,args.unet_model_path,is_load=True,is_freeze=True)
    lem_model      = load_lem_unet(args,args.seg_unet_path,is_load=True,is_freeze=True)
    dncn_model     = load_base_model(args,args.dncn_model_path,is_load=True,is_freeze=True)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)
    
    for epoch in range(start_epoch, args.num_epochs):

        scheduler.step(epoch)
        train_loss,train_time = train_epoch(args, epoch, model_re,model_conv1x1,dautomap_model,unet_model,lem_model,dncn_model,train_loader,optimizer,writer)
        dev_loss,dev_time = evaluate(args, epoch, model_re, model_conv1x1,dautomap_model,unet_model,lem_model,dncn_model,dev_loader, writer)
        visualize(args, epoch, model_re, model_conv1x1,dautomap_model,unet_model,lem_model,dncn_model,display_loader, writer)

        is_new_best = dev_loss < best_dev_loss
        best_dev_loss = min(best_dev_loss,dev_loss)
        save_model(args, args.exp_dir, epoch, model_re, model_conv1x1, optimizer,best_dev_loss,is_new_best)
        logging.info(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g}'
            f'DevLoss= {dev_loss:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s',
        )
    writer.close()


def create_arg_parser():

    parser = argparse.ArgumentParser(description='Train setup for MR recon U-Net')
    parser.add_argument('--seed',default=42,type=int,help='Seed for random number generators')
    parser.add_argument('--num-pools', type=int, default=4, help='Number of U-Net pooling layers')
    parser.add_argument('--drop-prob', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--num-chans', type=int, default=32, help='Number of U-Net channels')
    parser.add_argument('--batch-size', default=2, type=int,  help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lr-step-size', type=int, default=40,
                        help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help='Strength of weight decay regularization')
    parser.add_argument('--report-interval', type=int, default=100, help='Period of loss reporting')
    parser.add_argument('--data-parallel', action='store_true', 
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--exp-dir', type=pathlib.Path, default='checkpoints',
                        help='Path where model and results should be saved')
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to an existing checkpoint. Used along with "--resume"')
    parser.add_argument('--train-path',type=str,help='Path to train h5 files')
    parser.add_argument('--validation-path',type=str,help='Path to test h5 files')

    parser.add_argument('--acceleration_factor',type=str,help='acceleration factors')
    parser.add_argument('--dataset_type',type=str,help='cardiac,kirby')
    parser.add_argument('--unet_model_path',type=str,help='unet best model path')
    parser.add_argument('--dautomap_model_path',type=str,help='dautomap best model path')
    parser.add_argument('--seg_unet_path',type=str,help='unet model path')
    parser.add_argument('--dncn_model_path',type=str,help='unet model path')
    parser.add_argument('--usmask_path',type=str,help='us mask path')
    
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print (args)
    main(args)
