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
#from dataset import SliceData
from dataset_csv import SliceData
from models import DnCn
import torchvision
from torch import nn
from torch import optim
from tqdm import tqdm
from subsample import create_mask_for_mask_type
import transforms as T


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self, resolution, which_challenge, mask_func=None, use_seed=True):
        """
        Args:
            mask_func (common.subsample.MaskFunc): A function that can create a mask of
                appropriate shape.
            resolution (int): Resolution of the image.
            which_challenge (str): Either "singlecoil" or "multicoil" denoting the dataset.
            use_seed (bool): If true, this class computes a pseudo random number generator seed
                from the filename. This ensures that the same mask is used for all the slices of
                a given volume every time.
        """
        if which_challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.mask_func = mask_func
        self.resolution = resolution
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(self, kspace, mask, target, attrs, fname, slice):
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
                image (torch.Tensor): Zero-filled input image.
                target (torch.Tensor): Target image converted to a torch Tensor.
                mean (float): Mean value used for normalization.
                std (float): Standard deviation value used for normalization.
        """
        kspace = T.to_tensor(kspace)
        # Apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = T.apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace

        mask = mask.repeat(kspace.shape[0],1,1) # 640 is fixed for single coil 

        # Inverse Fourier Transform to get zero filled solution
        image = T.ifft2(masked_kspace)
        
        # Complex abs
        image_abs = T.complex_abs(image)
        image_abs_max = image_abs.max()
        
        # Image and kspace normalization
        image_norm = image / image_abs_max
        masked_kspace_norm = masked_kspace / image_abs_max
                
        # Crop input image to given resolution if larger
        smallest_width = min(self.resolution, image.shape[-2])
        smallest_height = min(self.resolution, image.shape[-3])
        
        if target is not None:
            smallest_width = min(smallest_width, target.shape[-1])
            smallest_height = min(smallest_height, target.shape[-2])

        crop_size = (smallest_height, smallest_width)
        crop_size_tensor = torch.Tensor(crop_size)
        
        image_norm_crop = T.complex_center_crop(image_norm, crop_size)
        kspace_norm_crop = T.fftshift(T.fft2(image_norm_crop))
        
        if target is not None:
            target = T.to_tensor(target)
            target_norm = target / image_abs_max
        else:
            target = torch.Tensor([0])
            
        return image_norm_crop, kspace_norm_crop, image_norm, masked_kspace_norm, target_norm, mask, image_abs_max, crop_size_tensor, fname, slice



def create_datasets(args):

    mask = create_mask_for_mask_type(args.mask_type, args.center_fractions, args.accelerations)

    transform = DataTransform(args.resolution, args.challenge, mask, use_seed=False)

    train_data = SliceData(
        root=args.train_path,
        csv_path=args.train_csv_path,
        transform=transform,
        sample_rate=args.sample_rate)

    dev_data = SliceData(
        root = args.validation_path,
        csv_path=args.valid_csv_path,
        transform=transform,
        sample_rate=args.sample_rate)

    return dev_data, train_data


def create_data_loaders(args):

    dev_data, train_data = create_datasets(args)   

    display_data = [dev_data[i] for i in range(0, len(dev_data), len(dev_data) // 16)]

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        #shuffle=True,
        #num_workers=1,
        #pin_memory=True,
    )
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        #num_workers=1,
        #pin_memory=True,
    )
    display_loader = DataLoader(
        dataset=display_data,
        batch_size=4,
        #num_workers=1,
        #pin_memory=True,
    )

    return train_loader, dev_loader, display_loader


def train_epoch(args, epoch, model,data_loader, optimizer, writer):
    
    model.train()
    avg_loss = 0.
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)
    #print ("Entering Train epoch")

    for iter, data in enumerate(tqdm(data_loader)):

        image_crop, kspace_crop, image, kspace, target, mask, _, crop_size_tensor, _, _ = data

        image_crop = image_crop.permute(0, 3, 1, 2).to(args.device)
        kspace_crop = kspace_crop.permute(0, 3, 1, 2).to(args.device)
        kspace = kspace.to(args.device)
        image = image.permute(0, 3, 1, 2).to(args.device)
        target = target.to(args.device)
        mask = mask.to(args.device)
       
        crop_size = (int(crop_size_tensor.numpy()[0,0]), int(crop_size_tensor.numpy()[0,1]))

        #print (image_crop.shape, kspace_crop.shape, image.shape, kspace.shape, mask.shape, crop_size_tensor.shape)
        output = model(image_crop, kspace_crop, image, kspace, mask, crop_size)
        output = T.complex_abs(output.permute(0,2,3,1))
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


def evaluate(args, epoch, model, data_loader, writer):

    model.eval()
    losses = []
    start = time.perf_counter()
    
    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):

            image_crop, kspace_crop, image, kspace, target, mask, _, crop_size_tensor, _, _ = data
    
            image_crop = image_crop.permute(0, 3, 1, 2).to(args.device)
            kspace_crop = kspace_crop.permute(0, 3, 1, 2).to(args.device)
            kspace = kspace.to(args.device)
            image = image.permute(0, 3, 1, 2).to(args.device)
            target = target.to(args.device)
            mask = mask.to(args.device)
           
            crop_size = (int(crop_size_tensor.numpy()[0,0]), int(crop_size_tensor.numpy()[0,1]))
    
            #print (image_crop.shape, kspace_crop.shape, image.shape, kspace.shape, mask.shape, crop_size_tensor.shape)
            output = model(image_crop, kspace_crop, image, kspace, mask, crop_size)
            output = T.complex_abs(output.permute(0,2,3,1))
    
            loss = F.l1_loss(output,target)
            
            losses.append(loss.item())
            #break
            
        writer.add_scalar('Dev_Loss',np.mean(losses),epoch)
       
    return np.mean(losses), time.perf_counter() - start


def visualize(args, epoch, model, data_loader, writer):
    
    def save_image(image, tag):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)

    model.eval()
    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):

            image_crop, kspace_crop, image, kspace, target, mask, abs_max, crop_size_tensor, _, _ = data
    
            image_crop = image_crop.permute(0, 3, 1, 2).to(args.device)
            kspace_crop = kspace_crop.permute(0, 3, 1, 2).to(args.device)
            kspace = kspace.to(args.device)
            image = image.permute(0, 3, 1, 2).to(args.device)
            target = target.to(args.device)
            mask = mask.to(args.device)
           
            crop_size = (int(crop_size_tensor.numpy()[0,0]), int(crop_size_tensor.numpy()[0,1]))
    
            #print (image_crop.shape, kspace_crop.shape, image.shape, kspace.shape, mask.shape, crop_size_tensor.shape)
            output = model(image_crop, kspace_crop, image, kspace, mask, crop_size)

            output = T.complex_abs(output.permute(0,2,3,1))
            image_crop_abs = T.complex_abs(image_crop.permute(0,2,3,1))

            save_image(image_crop_abs, 'Input')
            save_image(target, 'Target')
            save_image(output, 'Reconstruction')
            save_image(torch.abs(target.float() - output.float()), 'Error')

            break

def save_model(args, exp_dir, epoch, model, optimizer,best_dev_loss,is_new_best):

    out = torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir':exp_dir
        },
        f=exp_dir / 'model.pt'
    )

    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


def build_model(args):

    model = DnCn(args).to(args.device)    
   
    return model

def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model = build_model(args)

    if args.data_parallel:
        model = torch.nn.DataParallel(model)

    model.load_state_dict(checkpoint['model'])

    optimizer = build_optim(args, model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint, model, optimizer 


def build_optim(args, params):
    optimizer = torch.optim.Adam(params, args.lr, weight_decay=args.weight_decay)
    return optimizer


def main(args):
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    #writer = SummaryWriter(logdir=str(args.exp_dir / 'summary'))
    writer = SummaryWriter(log_dir=str(args.exp_dir / 'summary'))

    if args.resume:
        print('resuming model, batch_size', args.batch_size)
        #checkpoint, model, optimizer, disc, optimizerD = load_model(args, args.checkpoint)
        checkpoint, model, optimizer, disc, optimizerD = load_model(args.checkpoint)
        args = checkpoint['args']
        args.batch_size = 28
        best_dev_mse= checkpoint['best_dev_mse']
        best_dev_ssim = checkpoint['best_dev_mse']
        start_epoch = checkpoint['epoch']
        del checkpoint
    else:
        model = build_model(args)
        #print ("Model Built")
        if args.data_parallel:
            model = torch.nn.DataParallel(model)    
        optimizer = build_optim(args, model.parameters())
        #print ("Optmizer initialized")
        best_dev_loss = 1e9
        start_epoch = 0

    logging.info(args)
    logging.info(model)

    train_loader, dev_loader, display_loader = create_data_loaders(args)
    #print ("Dataloader initialized")
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)
    
    for epoch in range(start_epoch, args.num_epochs):

        scheduler.step(epoch)
        train_loss,train_time = train_epoch(args, epoch, model,train_loader,optimizer,writer)
        dev_loss,dev_time = evaluate(args, epoch, model, dev_loader, writer)
        #visualize(args, epoch, model, display_loader, writer)

        is_new_best = dev_loss < best_dev_loss
        best_dev_loss = min(best_dev_loss,dev_loss)
        save_model(args, args.exp_dir, epoch, model, optimizer,best_dev_loss,is_new_best)
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
    parser.add_argument('--train-csv-path',type=str,help='csv file containing the h5 file name')
    parser.add_argument('--valid-csv-path',type=str,help='csv file containing the h5 file name')

    parser.add_argument('--resolution', default=320, type=int, help='Resolution of images (brain \
                challenge expects resolution of 384, knee resolution expcts resolution of 320')
    parser.add_argument('--challenge', default = 'singlecoil', choices=['singlecoil', 'multicoil'], help='Which challenge')

    parser.add_argument('--mask-type', choices=['random', 'equispaced'], default='random',
                  help='The type of mask function to use')
    parser.add_argument('--accelerations', nargs='+', default=[4], type=int,
                  help='Ratio of k-space columns to be sampled. If multiple values are '
                       'provided, then one of those is chosen uniformly at random for '
                       'each volume.')
    parser.add_argument('--center-fractions', nargs='+', default=[0.08], type=float,
                          help='Fraction of low-frequency k-space columns to be sampled. Should '
                               'have the same length as accelerations')
    parser.add_argument('--sample-rate', type=float, default=0.25,
                          help='Fraction of total volumes to include')

    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print (args)
    main(args)
