#!/usr/bin/env python3

import torch
from torch import nn
from torch.nn.functional import kl_div
from torch.functional import norm
import torchvision
from torch.nn.functional import normalize
from torch.distributions.dirichlet import Dirichlet
import sys

# NUM_CLASSES = 5

''' Multi-Resolution Visual Representation Network for robot localization in the latent space '''
class CroScope(nn.Module):
    def __init__(self, model_tele = None, model_micro = None, device = 'cuda:0'):            
        super(CroScope, self).__init__()

        self._model_tele = model_tele
        self._model_micro = model_micro

        self.micro_only = False
        self.tele_only = False

        self.device = device

    def forward(self, tele_img=None, micro_imgs=None, pix_map=None):
        ''' reshape tensor '''

        tele_feature = None
        micro_feature = None
        if micro_imgs is not None:
            micro_imgs = micro_imgs.view(-1, 3, micro_imgs.shape[-2], micro_imgs.shape[-1])
            micro_feature = self._model_micro(micro_imgs) # (batchsize*microviews) * featuresize
            micro_feature = torch.softmax(micro_feature, 1)

        elif micro_imgs == None:
            ''' return tele feature '''            
            tele_feature = self._model_tele(tele_img)
            return torch.softmax(tele_feature['out'], 1)

        if tele_img is not None:
            tele_feature = self._model_tele(tele_img)['out']
            tele_feature = torch.softmax(tele_feature, 1)
        elif tele_img == None:
            ''' return micro feature '''
            return micro_feature

        # if self.training:
        ''' tele feature agg '''
        tele_feature_list = []
        # iter through each data (tele view) in the batch
        for i in range(pix_map.shape[0]):
            tele_feature_one_tele = []
            # iter through each micro view in each tele view
            for j in range(pix_map.shape[1]):
                ''' extract feature from pixel-wise analysis '''
                # print(pix_map[i, j, 0], pix_map[i, j, 1])
                tele_feature_one_tele.append(
                    tele_feature[i, :, pix_map[i, j, 0].long(), pix_map[i, j, 1].long()])
            tele_feature_list.append(torch.stack(tele_feature_one_tele))
        tele_feature_tensor = torch.stack(tele_feature_list) # batchsize*microviews*featuresize
        # print(tele_feature_tensor)
        ''' repeat each feature once, so have same size with augmented micro image (micor image number will double after aug) '''
        tele_feature_tensor_double = torch.repeat_interleave(
            tele_feature_tensor, repeats=2, dim=1)
        tele_feature_tensor_double = tele_feature_tensor_double.view(
            -1, tele_feature_tensor_double.shape[-1]) # (batchsize* 2*microviews)*featuresize

        num = micro_feature.shape[0]
        ''' Bhattacharya Coefficient '''
        logs = torch.matmul(torch.sqrt(tele_feature_tensor_double), torch.t(torch.sqrt(micro_feature)))

        labels = torch.arange(
            num, dtype=torch.long, device=torch.device(self.device))
        return logs, labels

    def micro_mode(self):
        self.tele_only = False
        self.micro_only = True        
        return

    def tele_mode(self):
        self.tele_only = True
        self.micro_only = False        
        return


class CroScope_ResResnet(CroScope):
    def __init__(self, device, num_cat):
        super(CroScope_ResResnet, self).__init__(device = device)       

        self.micro_only = False
        self.tele_only = False

        ''' Tele model '''
        self._model_tele = torchvision.models.segmentation.fcn_resnet50(
            pretrained=True, progress=True)
        # print(self.__model_tele)
        self._model_tele.classifier[4] = torch.nn.Conv2d(
            512, num_cat, kernel_size=(1, 1), stride=(1, 1))

        ''' Micro model '''
        self._model_micro = torchvision.models.resnet18(pretrained=True)
        self._model_micro.fc = nn.Sequential(
            nn.Linear(self._model_micro.fc.in_features, num_cat)
        )

    
    def set_tele_channels(self, modalities):
        channel = 0
        for item in modalities:
            channel = channel + int(item)
        self._model_tele.backbone.conv1 = torch.nn.Conv2d(channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        return


class CroScope_ResnetDeeplabv3(CroScope):
    def __init__(self, device, num_cat):
        super(CroScope_ResnetDeeplabv3, self).__init__(device = device)       

        self.micro_only = False
        self.tele_only = False

        ''' Tele model '''
        self._model_tele = torchvision.models.segmentation.deeplabv3_resnet50(
            pretrained=False, progress=True)
        self._model_tele.classifier[4] = nn.Conv2d(
            256, num_cat, kernel_size=(1, 1), stride=(1, 1))

        ''' Micro model '''
        self._model_micro = torchvision.models.resnet18(pretrained=True)
        self._model_micro.fc = nn.Sequential(
            nn.Linear(self._model_micro.fc.in_features, num_cat)
        )

    
    def set_tele_channels(self, modalities):
        channel = 0
        for item in modalities:
            channel = channel + int(item)
        self._model_tele.backbone.conv1 = nn.Conv2d(channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        return

def main():
    m2r_model = CroScope()
    return True


if __name__ == '__main__':
    main()
