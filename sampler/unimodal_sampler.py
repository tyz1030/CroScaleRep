# Parent class for telescope sampler and microscope sampler

import abc
import cv2 as cv
import numpy as np
from .sampler_utils import ImgAugmenter

class UniModalSampler():
    def __init__(self, sample_size = None) -> None:        
        self.img_auger = ImgAugmenter()
        if sample_size is not None:
            self.cal_sizes(sample_size)

        return

    def cal_sizes(self, sample_size):
        # 2D shape of sample in pixels [Height, Width]
        self.sample_size = np.array(sample_size)
        self.sample_size_half = np.ceil(self.sample_size/2.0)

        # pad size for image rotation in [Height, Width]
        self.pad_size = np.array(np.ceil(
            np.sqrt(np.sum((self.sample_size/2.0)**2)) - self.sample_size/2.0), dtype=np.int)
            
        self.half_size_with_pad = np.array(self.sample_size_half + self.pad_size, dtype=np.int)

        return

    def rand_pixel(self, pad_size = (0, 0)):
        idx_row = np.random.randint(pad_size[0], high=self.Height - pad_size[0])
        idx_col = np.random.randint(pad_size[1], high=self.Width - pad_size[1])
        return np.array([idx_row, idx_col])
    
    def has_white(self, img, threshold = 2):
        ''' arg: img has shape (H, W, C) '''
        small_res = 128
        small_img_size = small_res ** 2
        img = cv.resize(img, (small_res, small_res))

        ''' Rejection logic: If the sampled area has more than rej_perc percentage area is all white (not covered), then reject '''
        rej_perc = threshold/100.0  # if more than 2% is white, then resample
        num_channel = img.shape[2]
        if np.sum(np.sum(img, axis=2) == 255*num_channel)/small_img_size > rej_perc:
            print(self._scope + ': ' + 'This sample has white area more than ' + str(threshold) + '%. Resample...')
            return True
        return False

    @abc.abstractmethod
    def sample(self):
        raise NotImplementedError