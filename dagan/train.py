import sys
import logging
import pathlib
import random
import shutil
import time
import functools
import numpy as np
import torch
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch import nn
from torch.autograd import Variable
from torch import optim
from tqdm import tqdm
import argparse
from dataset import SliceData
from models import UnetModel,Discriminator,Vgg16


def create_datasets(args):

    train_data = SliceData(args.train_path,args.acceleration_factor,args.dataset_type)
    dev_data = SliceData(args.validation_path,args.acceleration_factor,args.dataset_type)

    return dev_data, train_data


def create_data_loaders(args):

    dev_data, train_data = create_datasets(args)

    display_data = [dev_data[i] for i in range(0, len(dev_data), len(dev_data) // 16)]

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=False,
    )

    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
    )
    display_loader = DataLoader(
        dataset=display_data,
        batch_size=16,
    )

    return train_loader, dev_loader, display_loader


def train_epoch(args, epoch, modelG, modelD, data_loader, optimizerG, optimizerD, writer, display_loader, exp_dir, vgg):

    modelG.train()
    modelD.train()

    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)

    #running_lossG = 0 Batch loss is used in graph and print statement 
    #running_lossD = 0

    criterionG = nn.MSELoss()
    criterionD = nn.BCEWithLogitsLoss()

    loss_fake = 0.

    adv_scale = 1
    img_scale = 15
    fft_scale = 0.1
    vgg_scale = 0.0025

    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)

    for iter, data in enumerate(tqdm(data_loader)):

        input,_,target = data

        input = input.unsqueeze(1).to(args.device)
        target = target.unsqueeze(1).to(args.device)

        input = input.float()
        target = target.float()

        batch_size = input.shape[0]

        outG = modelG(input)
        outG = outG + input # learning residual and adding to input and use this as loss         

        lossG_img = criterionG(outG,target) 

        features_target = vgg(target)
        features_outG   = vgg(outG)
        lossG_vgg       = F.mse_loss(features_target,features_outG)
 
        fft_target = torch.rfft(target,2,True,False)
        fft_outG   = torch.rfft(outG,2,True,False)
        lossG_fft   = F.mse_loss(fft_target,fft_outG)

        lossG = img_scale * lossG_img + fft_scale * lossG_fft + vgg_scale * lossG_vgg

        for param in modelD.parameters():
            param.requires_grad = True

        optimizerD.zero_grad()

        pred_fake = modelD(outG.detach())
        fake_label = torch.zeros(pred_fake.shape).to(args.device)
        lossD_fake = criterionD(pred_fake, fake_label)

        pred_real = modelD(target.float())
        real_label = torch.ones(pred_real.shape).to(args.device)
        lossD_real = criterionD(pred_real, real_label)

        lossD = (lossD_real + lossD_fake) * 0.5 
        lossD.backward()
        optimizerD.step()

        for param in modelD.parameters():
            param.requires_grad = False

        optimizerG.zero_grad()
        pred_fake = modelD(outG.detach())
        lossD_adversarial = criterionD(pred_fake, real_label)

        lossD_adversarial = (lossD_adversarial)*adv_scale
        lossGan = lossG + lossD_adversarial

        lossGan.backward()      
        optimizerG.step()

        writer.add_scalar('GenLoss', lossG.item(), global_step + iter)
        writer.add_scalar('DiscLoss', lossD.item(), global_step + iter)
        writer.add_scalar('lossD_fake', lossD_fake.item(), global_step+iter)
        writer.add_scalar('lossD_real', lossD_real.item(), global_step+iter)

        # Img, FFT, VGG loss 

        writer.add_scalar('ImageLoss', lossG_img.item(), global_step + iter)
        writer.add_scalar('FFTLoss', lossG_fft.item(), global_step + iter)
        writer.add_scalar('VGGLoss', lossG_vgg.item(), global_step + iter)
        

        #break

    return lossG.item(), lossD.item(), time.perf_counter() - start_epoch

def evaluate(args, epoch, model, data_loader, writer):

    model.eval()
    losses = []
    start = time.perf_counter()
    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):

            input,_,target = data
            input = input.unsqueeze(1).to(args.device)
            target = target.unsqueeze(1).to(args.device)
            
            input = input.float()
            target = target.float()
            output = model(input)
            output = output + input # learning residual and adding to input and use this as loss         

            loss = F.l1_loss(output,target)
            losses.append(loss.item())

            #break
        
        writer.add_scalar('Dev_Loss', np.mean(losses), epoch)
       
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
            input,_,target = data
            print("input: ", torch.min(input), torch.max(input))
            print("target: ", torch.min(target), torch.max(target))
            input = input.unsqueeze(1).to(args.device)
            target = target.unsqueeze(1).to(args.device)
            input  = input.float()
            output = model(input)
            output = output + input # learning residual and adding to input and use this as loss         
            
            print("Predicted: ", torch.min(output), torch.max(output))
            save_image(input, 'Input')
            save_image(target, 'Target')
            save_image(output, 'Reconstruction')
            save_image(torch.abs(target.float() - output.float()), 'Error')
            break

def save_model(args, exp_dir, epoch, modelG,optimizerG, modelD, optimizerD, best_dev_loss,is_new_best):

    out = torch.save(
        {
            'epoch': epoch,
            'args': args,
            'modelG': modelG.state_dict(),
            'optimizerG': optimizerG.state_dict(),
            'modelD':modelD.state_dict(),
            'optimizerD':optimizerD.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir':exp_dir
        },
        f=exp_dir / 'model.pt'
    )

    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


def build_model(args):

    model = UnetModel(
        in_chans=1,
        out_chans=1,
        chans=args.num_chans,
        num_pool_layers=args.num_pools,
        drop_prob=args.drop_prob
    ).to(args.device)

    return model

def build_discriminator(args):
    
    netD = Discriminator(input_nc=1).to(args.device)
    optimizerD = optim.SGD(netD.parameters(),lr=5e-3)
    
    return netD, optimizerD

def load_model(checkpoint_file):

    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    modelG = build_model(args)

    if args.data_parallel:
        modelG = torch.nn.DataParallel(modelG)

    modelG.load_state_dict(checkpoint['modelG'])

    optimizerG = build_optim(args, modelG.parameters())
    optimizerG.load_state_dict(checkpoint['optimizerG'])

    modelD,optimizerD = build_discriminator(args)

    if args.data_parallel:
        modelD = torch.nn.DataParallel(modelD)

    modelD.load_state_dict(checkpoint['modelD'])
    optimizerD.load_state_dict(checkpoint['optimizerD'])

    return checkpoint, modelG, optimizerG, modelD, optimizerD


def build_optim(args, params):
    optimizer = torch.optim.Adam(params, args.lr, weight_decay=args.weight_decay)
    return optimizer

def make_one_hot(labels, C=2):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.
    
    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size. 
        Each value is an integer representing correct classification.
    C : integer. 
        number of classes in labels.
    
    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    '''
    one_hot = torch.cuda.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
    target = one_hot.scatter_(1, labels.data, 1) 
        
    return target


def main(args):

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(args.exp_dir / 'summary'))
    vgg = Vgg16(requires_grad=False).to(args.device)


    if args.resume: 
        print('resuming model, batch_size', args.batch_size)
        checkpoint, modelG, optimizerG, modelD, optimizerD = load_model(args.checkpoint)
        bs = args.batch_size
        args = checkpoint['args']
        args.batch_size = bs
        start_epoch = checkpoint['epoch']
        del checkpoint

    else:

        modelG = build_model(args)
        modelD, optimizerD = build_discriminator(args)

        if args.data_parallel:
           modelG = torch.nn.DataParallel(modelG)
           modelD = torch.nn.DataParallel(modelD)

        optimizerG = build_optim(args, modelG.parameters())
        start_epoch = 0
        best_dev_loss = 1e9

    train_loader, dev_loader, display_loader = create_data_loaders(args)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizerG, args.lr_step_size, args.lr_gamma)

    for epoch in range(start_epoch, args.num_epochs+1):
        scheduler.step(epoch)
        train_lossG, train_lossD, train_time = train_epoch(args, epoch, modelG, modelD, train_loader, optimizerG , optimizerD, writer, display_loader, args.exp_dir,vgg)
        print ("Epoch {}".format(epoch))
        print ("Validation for epoch :{}".format(epoch))
        dev_loss, dev_time = evaluate(args, epoch, modelG, dev_loader, writer)
        
        print ("Visualization for epoch :{}".format(epoch))
        visualize(args, epoch, modelG, display_loader, writer)

        is_new_best = dev_loss < best_dev_loss
        best_dev_loss = min(best_dev_loss,dev_loss)

        save_model(args, args.exp_dir, epoch, modelG, optimizerG, modelD, optimizerD, best_dev_loss,is_new_best)
        logging.info(f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLossG = {train_lossG:.4g} TrainLossD = {train_lossD:.4g} 'f'DevNLL = {dev_loss:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s')
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
    parser.add_argument('--acceleration_factor',type=str,help='acceleration factors 2,2.5')
    parser.add_argument('--dataset_type',type=str,help='cardiac,kirby')

    
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
