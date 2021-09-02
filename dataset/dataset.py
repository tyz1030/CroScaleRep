#!/usr/bin/env python3

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
from skimage import io

from PIL import Image

from numpy import genfromtxt
import sys

sys.path.append("..")
from sampler.sampler_utils import ImgAugmenter

class CroScopeDataset(Dataset):
    """Dataset as pytorch data loader"""

    def __init__(self, args, transform_micro = transforms.Compose([transforms.ToTensor()]), transform_tele = transforms.Compose([transforms.ToTensor()])):
        """
        Args:
        data_path: path under which data are stored
        """
        self.data_path = args.dataset
        self.num_micro_view = args.num_microviews
        self.tele_modalities = args.tele_modalities
        self.aug_copy = args.aug_copy
        self.transform_micro = transform_micro
        self.transform_tele = transform_tele

        self.img_auger = ImgAugmenter()

    def __len__(self):
        return len(next(os.walk(self.data_path))[1])

    def __getitem__(self, idx):
        '''
        Return:
        tele_imgs_tensor: num of tele modalities * 
        micro_imgs_tensor: 
        pix_map_tensor: 
        '''

        ''' micro scope image '''
        micro_imgs_list = []
        for ii in range(self.num_micro_view):
            img_name = os.path.join(self.data_path, str(idx).zfill(5), 'micro' + str(ii).zfill(3) + ".png")
            image = io.imread(img_name)
            image_tensor = self.transform_micro(Image.fromarray(image))
            micro_imgs_list.append(image_tensor)

            if self.aug_copy:
                ''' image augment (rotate, color distortion, flip...)'''
                image_aug = self.img_auger.group_ops(image)
                image_tensor = self.transform_micro(Image.fromarray(image_aug))
                micro_imgs_list.append(image_tensor)
        micro_imgs_tensor = torch.stack(micro_imgs_list)

        ''' tele scope image '''
        tele_imgs_list = []
        for ii, element in enumerate(self.tele_modalities):
            if bool(element):
                img_name_tele = os.path.join(self.data_path, str(idx).zfill(5), 'tele' + str(ii) + '.png')            
                tele_img_tensor = self.transform_tele(io.imread(img_name_tele)) # grayscale will be transformed into 1*H*W, RGB will be 3*H*W
                for jj in range(tele_img_tensor.shape[0]):
                    tele_imgs_list.append(tele_img_tensor[jj, :, :])
        tele_imgs_tensor = torch.stack(tele_imgs_list)
        # print("tele")
        # print(tele_imgs_tensor.size())

        ''' pixel map, from micro image to location in tele scope image '''
        pix_file = os.path.join(self.data_path, str(idx).zfill(5), 'coordinate.csv')
        pix_map = genfromtxt(pix_file, delimiter=' ', skip_header=1)
        pix_map_tensor = torch.tensor(pix_map)[:self.num_micro_view, :]
        # print("pixel")
        # print(pix_map_tensor.size())

        return tele_imgs_tensor, micro_imgs_tensor, pix_map_tensor