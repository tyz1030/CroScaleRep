import numpy as np
import cv2 as cv
from .sampler_utils import GMapRequester
from .unimodal_sampler import UniModalSampler
from .sampler_utils import RandSelector, LineSelector, DenseSelector
import os

def pixel_selector_factory(mode, h, w, step=0):
    if mode =='random':
        return RandSelector(h, w)
    if mode =='line':
        return LineSelector(h, w, step = step)    
    if mode =='dense':
        return DenseSelector(h, w, step = step)

class MicroSampler(UniModalSampler):
    def __init__(self, data_size = (512, 512), sample_size = (224, 224), step = 8, mode = 'random') -> None:
        super(MicroSampler, self).__init__(sample_size = sample_size)
        self._scope = "Micro"
        self.pull_tele = None        
        self.data_size = data_size # [Height, Width], typically this is the sample size of the telescope image
        self.Height = data_size[0]
        self.Width = data_size[1]
        
        self.pixel_selector = pixel_selector_factory(mode, self.Height, self.Width, step = step)

    def register_tele_callback(self, fn_pull_tele):
        self.pull_tele = fn_pull_tele
        return




class MicroSamplerGMap(MicroSampler):
    def __init__(self, api_key = None, secret = None, step = 8, data_size = (512, 512), sample_size = (224, 224), hifi = True, mode = 'random') -> None:
        super(MicroSamplerGMap, self).__init__(data_size = data_size, sample_size = sample_size, step = step, mode = mode)
        self.__sampler_type = 'Micro-Scope'
        self.__source_type = 'Google Map Static API'
        request_size = sample_size if not hifi else (640, 640) # hardcode due to api limit resolution
        self.gmap_requester = GMapRequester(api_key, secret = secret, size = request_size)
        return

    def sample(self, subpath, idx_micro):
        # STEP 1: generate random position as center of sample
        pix_idx = self.pixel_selector.select_pixel()
        coord_u = 0.5+pix_idx[0]
        coord_v = 0.5+pix_idx[1] # in unit of pixels, 0.5 to be centered at certain pixel
        # STEP 2: query geo coordinate of position
        geo_coord = self.pull_tele('pix2coord', pixel = [coord_u, coord_v])
        # STEP 3: request micro image from google
        img = self.gmap_requester.request_from_url(geo_coord) # by default going to be 1280*1280
        # STEP 4: data augmentation (rotation)
        img_rot, rot_angle, _ = self.img_auger.rand_rotate(img)
        pad = 640-448 # magic number again due to api size. 640 is half size of map api return
        img = img_rot[pad:1280-pad, pad:1280-pad, :] # cut final sample from patch, with hifi = True this will be 896*896
        img = cv.resize(img, (self.sample_size[0], self.sample_size[1])) # resize to desired size            
        return img, pix_idx, rot_angle

class MicroSamplerTiff(MicroSampler):
    def __init__(self, white_check = False, mode = 'random', step = 8) -> None:
        super(MicroSamplerTiff, self).__init__()
        self.white_check = white_check
        self.pixel_selector = pixel_selector_factory(mode, self.Height, self.Width, step = step)

    def sample(self, subpath, idx_micro):        
        while(True):
            # STEP 1: generate random position as center of sample
            pix_idx = self.pixel_selector.select_pixel()  
            coord_u = 0.5+pix_idx[0]
            coord_v = 0.5+pix_idx[1] # in unit of pixels, 0.5 to be centered at certain pixel
            # STEP 2: get the corresponding pixel in the superres map
            half_s = self.half_size_with_pad
            _, samp_padded = self.pull_tele('pix2superres', pixel = [coord_u, coord_v], half_s = half_s)
            # STEP 3: data augmentation (rotation)
            img_rot, rot_angle, _ = self.img_auger.rand_rotate(samp_padded)
            # STEP 4: cut final sample from rotated patch
            ps = self.pad_size
            sample = img_rot[ps[0]:half_s[0] *
                                2-ps[0], ps[1]:half_s[1]*2-ps[1], :]
            sample = sample[0:self.sample_size[0], 0:self.sample_size[1], :]            
            # STEP 5: check if a valid a sample
            if self.white_check and self.has_white(sample, threshold = self.white_check):
                continue
            else:
                break
        return sample, pix_idx, rot_angle

class MicroSamplerGMapChip(MicroSamplerGMap):
    def __init__(self, api_key = None, secret = None, step = 8, data_size = (512, 512), sample_size = (224, 224), hifi = True, mode = 'random') -> None:
        super(MicroSamplerGMapChip, self).__init__(api_key = api_key, secret = secret, step = step, data_size = data_size, sample_size = sample_size, hifi = hifi, mode = mode)
        return

    def sample(self, subpath, idx_micro):
        # STEP 1: generate random position as center of sample
        pix_idx = self.pixel_selector.select_pixel()
        coord_u = 0.5+pix_idx[0]
        coord_v = 0.5+pix_idx[1] # in unit of pixels, 0.5 to be centered at certain pixel
        # STEP 2: query geo coordinate of position
        geo_coord = self.pull_tele('pix2coord', pixel = [coord_u, coord_v])
        # STEP 3: request micro image from google
        img = self.gmap_requester.request_from_url(geo_coord) # by default going to be 1280*1280
        # STEP 4: data augmentation (rotation)
        img_rot, rot_angle, _ = self.img_auger.rand_rotate(img)
        pad = 640-448 # magic number again due to api size. 640 is half size of map api return
        img = img_rot[pad:1280-pad, pad:1280-pad, :] # cut final sample from patch, with hifi = True this will be 896*896
        img = cv.resize(img, (self.sample_size[0], self.sample_size[1])) # resize to desired size 
        
        # STEP 2.5: pull a low res chip
        _, chip = self.pull_tele('pix2superres', pixel = [coord_u, coord_v], half_s = (16, 16))
        chip = cv.resize(chip, (32, 32))
        filename_chip = "randchip" + str(idx_micro).zfill(3) + ".png"
        fullname_chip = os.path.join(subpath, filename_chip)
        cv.imwrite(fullname_chip, chip[:, :, :])
        return img, pix_idx, rot_angle
        
class MicroSamplerTiffChip(MicroSamplerTiff):
    def __init__(self, white_check = False, mode = 'random', step = 8) -> None:
        super(MicroSamplerTiffChip, self).__init__(white_check = white_check, mode = mode, step = step)

    def sample(self, subpath, idx_micro):        
        while(True):
            # STEP 1: generate random position as center of sample
            pix_idx = self.pixel_selector.select_pixel()  
            coord_u = 0.5+pix_idx[0]
            coord_v = 0.5+pix_idx[1] # in unit of pixels, 0.5 to be centered at certain pixel
            # STEP 2: get the corresponding pixel in the superres map
            half_s = self.half_size_with_pad
            _, samp_padded = self.pull_tele('pix2superres', pixel = [coord_u, coord_v], half_s = half_s)
            # STEP 3: data augmentation (rotation)
            img_rot, rot_angle, _ = self.img_auger.rand_rotate(samp_padded)
            # STEP 4: cut final sample from rotated patch
            ps = self.pad_size
            sample = img_rot[ps[0]:half_s[0] *
                                2-ps[0], ps[1]:half_s[1]*2-ps[1], :]
            sample = sample[0:self.sample_size[0], 0:self.sample_size[1], :]            
            # STEP 5: check if a valid a sample
            if self.white_check and self.has_white(sample, threshold = self.white_check):
                continue
            else:
                break
        
        # STEP 2.5: pull a low res tile
        _, chip = self.pull_tele('pix2superres', pixel = [coord_u, coord_v], half_s = (160, 160))
        chip = cv.resize(chip, (32, 32))
        filename_chip = "randchip" + str(idx_micro).zfill(3) + ".png"
        fullname_chip = os.path.join(subpath, filename_chip)
        cv.imwrite(fullname_chip, chip)

        return sample, pix_idx, rot_angle
