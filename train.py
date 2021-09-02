#!/usr/bin/env python3

import numpy as np
import torch
from torch import nn
import sys
from subprocess import call
import os
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as transforms

from dataset.dataset import CroScopeDataset
from model.model import CroScope_ResResnet, CroScope_ResnetDeeplabv3

import time

RAND_SEED = 906


def print_system():
    print('__Python VERSION:', sys.version)
    print('__pyTorch VERSION:', torch.__version__)
    print('__CUDA VERSION')
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__Devices')
    call(["nvidia-smi", "--format=csv",
          "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    print('Active CUDA Device: GPU', torch.cuda.current_device())
    print('Available devices ', torch.cuda.device_count())
    print('Current cuda device ', torch.cuda.current_device())
    # print(torch.cuda.get_device_name(1))
    return


def train(train_loader, val_loader, model, criterion, optimizer, scheduler, args):
    writer = SummaryWriter('runs/scott300epoch')
    num_epoch = args.epoch
    ws = args.workspace

    for epoch in range(num_epoch):  # loop over the dataset multiple times
        epoch_start = time.time()
        print("Epoch " + str(epoch+1)+'/'+str(num_epoch))

        ''' training '''
        training_loss = 0.0
        model.train()
        for tele_imgs, micro_imgs, pix_map in train_loader:
            tele_imgs, micro_imgs = tele_imgs.to(args.gpu), micro_imgs.to(args.gpu)
            optimizer.zero_grad()
            logs, labels = model(tele_img=tele_imgs,
                                 micro_imgs=micro_imgs, pix_map=pix_map)
            loss = criterion(logs/args.temp_param, labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()*tele_imgs.size(0)
        print("training loss: " + str(training_loss))
        writer.add_scalar('training loss', training_loss, epoch)

        ''' validation '''
        validation_loss = 0.0
        model.eval()
        for tele_imgs, micro_imgs, pix_map in val_loader:
            tele_imgs, micro_imgs = tele_imgs.to(args.gpu), micro_imgs.to(args.gpu)
            logs, labels = model(tele_img=tele_imgs,
                                 micro_imgs=micro_imgs, pix_map=pix_map)
            loss = criterion(logs/args.temp_param, labels)
            validation_loss += loss.item()*tele_imgs.size(0)
        scheduler.step(validation_loss)
        print("validation loss: " + str(validation_loss))
        writer.add_scalar('validation loss', validation_loss, epoch)
        
        if epoch%10 == 9:
            torch.save(model.state_dict(), ws+'/model' + str(epoch).zfill(3) +'.pth')
        epoch_end = time.time()
        print('epoch consume: '+str(epoch_end-epoch_start))

    torch.save(model.state_dict(), ws+"/model.pth")
    return


def config_parser():
    import argparse
    parser = argparse.ArgumentParser(
        description='Train Cross scope network')
    parser.add_argument('--dataset', type=str,
                        help='path to dataset (will split into train and val)')
    parser.add_argument('--num_train', type=int, help='size of training set')
    parser.add_argument('--num_val', type=int, help='size of validation set')
    parser.add_argument('--epoch', type=int, help='number of epoch')
    parser.add_argument('--num_microviews', default=16,
                        type=int, help='number of micro views')
    parser.add_argument('--batch_size', default=4,
                        type=int, help='number of batch size')
    parser.add_argument('--workspace', type=str,
                        help='output path for trained model and logs')
    parser.add_argument('--tele_modalities', type=str,
                        help='enabled telescope modalities')
    parser.add_argument('--aug_copy', default=1,
                        type=int, help=' ')
    parser.add_argument('--gpu', default='cuda:0',
                        type=str, help=' ')
    parser.add_argument('--temp_param', default=1.0,
                        type=float, help=' ')
    parser.add_argument('--lr', default=0.001,
                        type=float, help=' ')
    parser.add_argument('--category', default=5,
                        type=int, help=' ')
    parser.add_argument('--backbone', default='fcn',
                        type=str, help=' ')
    return parser

def check_ws(ws):
    if ws is None:
        sys.exit('Missing workspace argument. Exited.')
    if not os.path.exists(ws):
        os.makedirs(ws)
    elif os.path.isdir(ws):
        if len(os.listdir(ws)):
            sys.exit('Workspace directory is not empty. Exited.')

def flush_config(args):
    ''' log arguments'''
    ws = args.workspace
    config_file = os.path.join(ws, "config.yaml")
    file_opened = open(config_file, "w")
    for key in args.__dict__:
        if args.__dict__[key] is not None:
            file_opened.write(key+": " + str(args.__dict__[key])+"\n")
    file_opened.close()
    return

def main():
    # print_system()
    parser = config_parser()
    args = parser.parse_args()
    
    check_ws(args.workspace)
    flush_config(args)

    device = args.gpu

    micro_tsfm = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)  # not strengthened
            ], p=0.8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    micro_tsfm = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    dataset = CroScopeDataset(args, transform_micro=micro_tsfm)
    train_set, val_set = torch.utils.data.random_split(
        dataset, [args.num_train, args.num_val], generator=torch.Generator().manual_seed(RAND_SEED))
    train_dataloader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(
        val_set, batch_size=1, shuffle=True, num_workers=4, drop_last=True)

    if torch.cuda.is_available():
        print("cuda avaliable")

    if args.backbone == 'fcn':
        model_croscope = CroScope_ResResnet(device, args.category)
    elif args.backbone == 'deeplab':
        model_croscope = CroScope_ResnetDeeplabv3(device, args.category)
    model_croscope.set_tele_channels(args.tele_modalities)
    model_croscope = model_croscope.to(device)

    criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
    optim = torch.optim.SGD(model_croscope.parameters(), lr=args.lr, momentum=0.9)
    scheduler = ReduceLROnPlateau(optim, 'min')
    train(train_dataloader, val_dataloader,
          model_croscope, criterion, optim, scheduler, args)
    return True


if __name__ == '__main__':
    main()