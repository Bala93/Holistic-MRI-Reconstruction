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
from architecture import network
import torchvision
from torch import nn
from torch.autograd import Variable
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

    def __call__(self, kspace, sensitivity, target, mask, data_fname, sensitivity_fname):
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
        sensitivity = T.to_tensor(sensitivity)

        # Apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = T.apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace

        ## Check the dimension and change accordingly
        mask = mask.repeat(kspace.shape[0],kspace.shape[1],1,2) 

        ## normalization factor 
        masked_image = T.ifft2(masked_kspace)
        masked_image_rss = T.root_sum_of_squares(T.complex_abs(masked_image),dim=0)
        masked_image_rss_max = masked_image_rss.max()

        ## kspace normalization 
        masked_kspace_norm = masked_kspace / masked_image_rss_max

        # Inverse Fourier Transform to get zero filled solution
        masked_image_norm = T.ifft2(masked_kspace_norm)
        
        ## combine channel and sensitivity map
        masked_image_norm_reduce = T.complex_mul(masked_image_norm, T.complex_conj(sensitivity)).sum(dim=0) ## Fu network input 
        masked_image_norm_reduce_crop = T.complex_center_crop(masked_image_norm_reduce, (320,320)) ## II networt input
        masked_kspace_norm_reduce_crop = T.fftshift(T.fft2(masked_image_norm_reduce_crop)) ## KI network input TODO: Check fftshift 

        if target is not None:

            kspace_norm = kspace / masked_image_rss_max
            image_norm = T.ifft2(kspace_norm)
            image_norm_reduce = T.complex_mul(image_norm, T.complex_conj(sensitivity)).sum(dim=0)
            image_norm_expand = T.complex_mul(image_norm_reduce, sensitivity)
            target_sense_rss = T.root_sum_of_squares(T.complex_abs(T.complex_center_crop(image_norm_expand,(320,320))),dim=0)
            target_rss = target / masked_image_rss_max

        else:

            target_sense_rss = torch.Tensor([0])
            target_rss = torch.Tensor([0])
 
        #return masked_image_norm_reduce_crop, masked_kspace_norm_reduce_crop, masked_image_norm_reduce, sensitivity, mask, masked_kspace_norm, target_sense_rss, target_rss, masked_image_rss_max
        return masked_image_norm_reduce_crop, masked_kspace_norm_reduce_crop, masked_image_norm_reduce, sensitivity, mask, masked_kspace_norm, image_norm_reduce, target_rss, masked_image_rss_max


def create_datasets(args):

    mask = create_mask_for_mask_type(args.mask_type, args.center_fractions, args.accelerations)

    transform = DataTransform(args.resolution, args.challenge, mask, use_seed=False)

    train_data = SliceData(
        root=args.train_path,
        csv_path=args.train_csv_path,
        transform=transform,
        challenge=args.challenge,
        sample_rate=args.sample_rate)

    dev_data = SliceData(
        root = args.validation_path,
        csv_path=args.valid_csv_path,
        transform=transform,
        challenge=args.challenge,
        sample_rate=args.sample_rate)

    return dev_data, train_data


def create_data_loaders(args):

    dev_data, train_data = create_datasets(args)   

    display_data = [dev_data[i] for i in range(0, len(dev_data), len(dev_data) // 16 )]

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        #shuffle=True,
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
        batch_size=1,
        #num_workers=64,
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

        img_und_crop, img_und_kspace, img_und, sensitivity, masks, rawdata_und, img_gt, _, _ = data

        #print (img_und_crop.shape, img_und_kspace.shape, img_und.shape, sensitivity.shape, masks.shape, rawdata_und.shape, img_gt.shape)

        img_gt  = img_gt.to(args.device)
        rawdata_und = rawdata_und.to(args.device)
        masks = masks.to(args.device)

        img_und_crop = img_und_crop.to(args.device)
        img_und = img_und.to(args.device)
        img_und_kspace = img_und_kspace.to(args.device)
        sensitivity = sensitivity.to(args.device)
        
        output = model(img_und_crop, img_und_kspace, img_und, rawdata_und,masks,sensitivity,(320,320))
       
        output_expand = T.complex_mul(output, sensitivity)
        output_expand_rss = T.root_sum_of_squares(T.complex_abs(T.complex_center_crop(output_expand,(320,320))),dim=1)

        #print (torch.sum(output), torch.sum(output_expand), torch.sum(output_expand_rss), torch.sum(img_gt))
        #print (output.shape, output_expand.shape, output_expand_rss.shape,img_gt.shape)

        #loss = F.mse_loss(output_expand_rss, img_gt)
        loss = F.mse_loss(output, img_gt)

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

    losses_sense = []
    losses_target = []

    start = time.perf_counter()
    
    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):

            img_und_crop, img_und_kspace, img_und, sensitivity, masks, rawdata_und, img_gt, target, _ = data
    
            #print (img_und_crop.shape, img_und_kspace.shape, img_und.shape, sensitivity.shape, masks.shape, rawdata_und.shape, img_gt.shape)
    
            img_gt  = img_gt.to(args.device)
            rawdata_und = rawdata_und.to(args.device)
            masks = masks.to(args.device)
    
            img_und_crop = img_und_crop.to(args.device)
            img_und = img_und.to(args.device)
            img_und_kspace = img_und_kspace.to(args.device)
            sensitivity = sensitivity.to(args.device)
            target = target.to(args.device)
            
            output = model(img_und_crop, img_und_kspace, img_und, rawdata_und,masks,sensitivity,(320,320))
           
            output_expand = T.complex_mul(output, sensitivity)
            output_expand_rss = T.root_sum_of_squares(T.complex_abs(T.complex_center_crop(output_expand,(320,320))),dim=1)

            #print (torch.sum(output_expand_rss), torch.sum(img_gt))

            loss_sense  = F.mse_loss(output, img_gt)
            loss_target = F.mse_loss(output_expand_rss, target)
            #print (loss_sense, loss_target)
    
            losses_sense.append(loss_sense.item())
            losses_target.append(loss_target.item())
            #break
            
        writer.add_scalar('Dev_Loss_sense',np.mean(losses_sense),epoch)
        writer.add_scalar('Dev_Loss_target',np.mean(losses_target),epoch)
       
    return np.mean(losses_sense), time.perf_counter() - start


def visualize(args, epoch, model, data_loader, writer):
    
    def save_image(image, tag):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)

    model.eval()
    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):

            #print (img_und_abs.shape,img_gt_abs.shape,output_abs.shape)

            img_und_crop, img_und_kspace, img_und, sensitivity, masks, rawdata_und, img_gt, target, _ = data
    
            #print (img_und_crop.shape, img_und_kspace.shape, img_und.shape, sensitivity.shape, masks.shape, rawdata_und.shape, img_gt.shape)
    
            img_gt  = img_gt.to(args.device)
            target = target.to(args.device)
            rawdata_und = rawdata_und.to(args.device)
            masks = masks.to(args.device)
    
            img_und_crop = img_und_crop.to(args.device)
            img_und = img_und.to(args.device)
            img_und_kspace = img_und_kspace.to(args.device)
            sensitivity = sensitivity.to(args.device)
            
            output = model(img_und_crop, img_und_kspace, img_und, rawdata_und,masks,sensitivity,(320,320))
           
            output_expand = T.complex_mul(output, sensitivity)
            output_expand_rss = T.root_sum_of_squares(T.complex_abs(T.complex_center_crop(output_expand,(320,320))),dim=1)

            #print (img_gt.shape, output_expand_rss.shape)

            save_image(target, 'Target')
            save_image(output_expand_rss, 'Reconstruction')
            save_image(torch.abs(output_expand_rss - target), 'Error')

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

    dccoeff = 0.1
    wacoeff = 0.1
    cascade = 3

    model = network(dccoeff,wacoeff,cascade).to(args.device)

    return model


def build_optim(args, params):
    print("args.lr: ",args.lr)
    print("args.weight_decay: ",args.weight_decay)
    print("params: ",params)
    optimizer = torch.optim.Adam(params, args.lr, weight_decay=args.weight_decay)
    print(optimizer)
    return optimizer


def main(args):

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(args.exp_dir / 'summary'))

    model = build_model(args)

    if args.data_parallel:
        model = torch.nn.DataParallel(model)    

    optimizer = build_optim(args, model.parameters())
    print ("Optmizer initialized")
    best_dev_loss = 1e9
    start_epoch = 0

    logging.info(args)
    logging.info(model)

    train_loader, dev_loader, display_loader = create_data_loaders(args)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)
    
    for epoch in range(start_epoch, args.num_epochs):

        scheduler.step(epoch)
        train_loss,train_time = train_epoch(args, epoch, model, train_loader,optimizer,writer)
        dev_loss,dev_time = evaluate(args, epoch, model, dev_loader, writer)
        visualize(args, epoch, model, display_loader, writer)

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
