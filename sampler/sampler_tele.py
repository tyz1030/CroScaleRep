import os
import glob
from attr import attr
import yaml
import cv2 as cv
import numpy as np
import sys
from .sampler_utils import ImgAugmenter
from .unimodal_sampler import UniModalSampler
import abc


class TeleSampler(UniModalSampler):
    def __init__(self, args, sample_size=(512, 512)) -> None:
        super(TeleSampler, self).__init__(sample_size=sample_size)
        self._scope = "Tele"
        self.img_auger = ImgAugmenter()

        # numpy array, has dimension HEIGHT*WIDTH*CHANNEL
        self.load_data(args)

        # rotation matrix from rotated sample coordinate to world coordinate
        self.R_s_w = np.eye(2)
        # rotation matrix from world coordinate to rotated sample coordinate
        self.R_w_s = np.eye(2)

        self.coordinate = None  # A dictionary, have range of latitude and longtitude

        self.scale_down = args.scale_down

        # np.random.seed(900)
        return

    def get_sample_attribute(self, attribute, **kwargs):
        if attribute == 'R_s_w':
            # Rotation matrix from sample to world frame
            return self.R_s_w
        if attribute == 'coordinate':
            return self.coordinate
        if attribute == 'pix2coord':  # pixel coordinate to geo coordinate
            pix = kwargs.get('pixel', None)
            return self.pix_to_coordinate(pix) if pix is not None else False
        if attribute == 'pix2superres':  # pixel coordinate to pixel coordinate in super res image
            pix = kwargs.get('pixel', None)
            half_padded_size = kwargs.get('half_s', None)
            return self.pix_to_superres(pix, half_padded_size) if pix is not None else False

    @abc.abstractmethod
    def load_data(self, data_path): raise NotImplementedError

    @abc.abstractmethod
    def sample(self): raise NotImplementedError

    @abc.abstractmethod
    def pix_to_coordinate(self, pixel_coordinate):
        '''
        Input: pixel coordinate in sample tile
        Return: geo coordinate in longtitude and latitude
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def pix_to_superres(self, pixel_coordinate):
        '''
        Input: pixel coordinate in sample tile
        Return: geo coordinate in longtitude and latitude
        '''
        raise NotImplementedError


class TeleSamplerTiff(TeleSampler):
    def __init__(self, args, sample_size=(512, 512)) -> None:
        super(TeleSamplerTiff, self).__init__(
            args, sample_size=sample_size)
        self.white_check = args.white_check_tele
        return

    def load_data(self, args):
        tif_path = os.path.join(args.data, '*')+args.data_type
        data_files = glob.glob(tif_path)
        if len(data_files) == 0:
            sys.exit(args.data_type + ' file not found. Exited.')
        elif len(data_files) >= 2 and args.data_type == '.vrt':
            sys.exit('.vrt file not found. Exited.')

        import rasterio
        data = None
        for tele_tif in data_files:
            src = rasterio.open(tele_tif)
            # Dimension: Channel * Height * Width
            data = src.read() if data is None else np.concatenate((data, src.read()), axis=0)
        if not np.issubdtype(data.dtype, np.unsignedinteger):
            data = data - np.amin(data, axis=(1, 2), keepdims=True)
            data = np.round(
                255 * data/np.amax(data, axis=(1, 2), keepdims=True))
        # Dimension: Height * Width * Channel
        self.data = np.transpose(data, (1, 2, 0))
        if args.flip_axis is not None:
            # the tiff image will be upside down if not flip
            self.data = np.flip(self.data, args.flip_axis)
        if args.flip_channels:
            self.data = np.flip(self.data, 2)
        if args.scale_down:
            self.superres = np.copy(self.data)
            size = (int(self.data.shape[1]/args.scale_down),
                    int(self.data.shape[0]/args.scale_down))
            self.data = cv.resize(self.data, size)
        self.Height = self.data.shape[0]
        self.Width = self.data.shape[1]
        self.Channel = self.data.shape[2]

        # Load geo coordinate of tiff
        coordinate_file = os.path.join(args.data, "coordinate.yaml")
        if os.path.exists(coordinate_file):
            self.coord = yaml.load(open(coordinate_file))
            lat_diff = self.coord['latitude'][1] - \
                self.coord['latitude'][0]
            self.lat2hei = lat_diff/self.Height  # ratio of latitude to height
            lon_diff = self.coord['longtitude'][1] - self.coord['longtitude'][0] if self.coord['longtitude'][
                1] > self.coord['longtitude'][0] else 360 + self.coord['longtitude'][1] - self.coord['longtitude'][0]
            self.lon2wid = lon_diff/self.Width  # ratio of longtitude to width
        else:
            print("Geo coordinate file does not exist. Skipped loading...")
        return

    def sample(self, subpath):
        while(True):
            ps = self.pad_size  # ps[0] is in Height, ps[1] is in Width
            half_s = self.half_size_with_pad

            # STEP 1: generate random position as center of sample
            pix_idx = self.rand_pixel(pad_size=half_s)
            # pix_idx = np.array([2000, 3000]) # hardcode fixed point sample
            self.pix_center = pix_idx

            if hasattr(self, 'coord'):
                lat = self.coord['latitude'][1] - pix_idx[0] * self.lat2hei
                lon = self.coord['longtitude'][0] + pix_idx[1] * self.lon2wid
                self.geo_center = np.array([lat, lon])

            # STEP 2: cut a patch for rotation
            samp_padded = self.data[pix_idx[0] -
                                    half_s[0]:pix_idx[0]+half_s[0], pix_idx[1]-half_s[1]:pix_idx[1]+half_s[1], :]

            # STEP 3: generate random rotation and rotate patch
            samp_padded, rot_angle, self.rot_mat = self.img_auger.rand_rotate(
                samp_padded)
            # samp_padded, rot_angle, self.rot_mat = self.img_auger.fixed_rotate(
            #     samp_padded)
            # STEP 4: cut final sample from rotated patch
            sample = samp_padded[ps[0]:half_s[0] *
                                 2-ps[0], ps[1]:half_s[1]*2-ps[1], :]
            sample = sample[0:self.sample_size[0], 0:self.sample_size[1], :]
            # STEP 5: check if a valid a sample
            if self.white_check and self.has_white(sample, threshold=self.white_check):
                continue
            else:
                break
        return sample, pix_idx, rot_angle

    def pix_to_coordinate(self, pixel_coordinate):
        # STEP 0: u-v coordinate to image plane coordinate
        # ------>v              ------> x
        # |                     |
        # |             =====>  |
        # v u                   v y
        img_coord = np.array((pixel_coordinate[1], pixel_coordinate[0]))
        half_img_size = np.array(self.sample_size[1], self.sample_size[0])/2.0

        # STEP 1: Rotate pixel back
        pixel = np.matmul(np.transpose(
            self.rot_mat[:2, :2]), img_coord - half_img_size)

        # STEP 2: Interpolate to geo coordinate
        lon_offset = pixel[0]*self.lon2wid
        lat_offset = -pixel[1]*self.lat2hei
        geo_coordinate = self.geo_center + np.array([lat_offset, lon_offset])

        return geo_coordinate

    def pix_to_superres(self, pixel_coordinate, half_padded_size=None):
        img_coord = np.array((pixel_coordinate[1], pixel_coordinate[0]))
        half_img_size = np.array(self.sample_size[1], self.sample_size[0])/2.0

        # STEP 1: Rotate pixel back
        pixel = np.matmul(np.transpose(
            self.rot_mat[:2, :2]), img_coord - half_img_size)
        pix_superres = self.pix_center + np.array((pixel[1], pixel[0]))
        pix_superres = np.around(pix_superres * self.scale_down).astype(int)
        # STEP 2: pull a tile from superres map
        samp_padded = None
        if half_padded_size is not None:
            samp_padded = self.superres[pix_superres[0] - half_padded_size[0]:pix_superres[0] +
                                        half_padded_size[0], pix_superres[1]-half_padded_size[1]:pix_superres[1]+half_padded_size[1], :]
        return pix_superres, samp_padded


class TeleSamplerTiffDense(TeleSamplerTiff):
    def __init__(self, args, sample_size=(512, 512)) -> None:
        super(TeleSamplerTiffDense, self).__init__(
            args, sample_size=sample_size)
        self.tile_half_s = 16  # hard code tile size 32*32
        return

    def sample(self, subpath):
        while(True):
            ps = self.pad_size  # ps[0] is in Height, ps[1] is in Width
            half_s = self.half_size_with_pad

            # STEP 1: generate random position as center of sample
            pix_idx = self.rand_pixel(pad_size=half_s)
            # pix_idx = np.array([2000, 3000]) # hardcode fixed point sample
            self.pix_center = pix_idx

            if hasattr(self, 'coord'):
                lat = self.coord['latitude'][1] - pix_idx[0] * self.lat2hei
                lon = self.coord['longtitude'][0] + pix_idx[1] * self.lon2wid
                self.geo_center = np.array([lat, lon])

            # STEP 2: cut a patch for rotation
            samp_padded = self.data[pix_idx[0] -
                                    half_s[0]:pix_idx[0]+half_s[0], pix_idx[1]-half_s[1]:pix_idx[1]+half_s[1], :]

            # STEP 3: generate random rotation and rotate patch
            samp_padded, rot_angle, self.rot_mat = self.img_auger.rand_rotate(
                samp_padded)
            # samp_padded, rot_angle, self.rot_mat = self.img_auger.fixed_rotate(
            #     samp_padded)

            # STEP 4: cut final sample from rotated patch
            sample = samp_padded[ps[0]:half_s[0] *
                                 2-ps[0], ps[1]:half_s[1]*2-ps[1], :]
            sample = sample[0:self.sample_size[0], 0:self.sample_size[1], :]
            # STEP 5: check if a valid a sample
            if self.white_check and self.has_white(sample, threshold=self.white_check):
                continue
            else:
                break

        # STEP 4+: cut tiles for comparison
        for ii in range(self.sample_size[0]):  # rows
            row_path = subpath+"/row"+str(ii).zfill(4)
            os.mkdir(row_path)
            # print(row_path)
            for jj in range(self.sample_size[1]):  # cols
                tile = samp_padded[ps[0]-self.tile_half_s+ii:ps[0]+self.tile_half_s +
                                   ii, ps[1]-self.tile_half_s+jj:ps[1]+self.tile_half_s+jj, :]
                filename = row_path+"/"+str(jj).zfill(4)+".png"
                cv.imwrite(filename, tile)
                pass

        return sample, pix_idx, rot_angle


DATA = '/home/tyz/geo_loc_data/kempten/raw_telescope'


def main():
    # test code
    tele_sampler = TeleSamplerTiff(DATA)
    return


if __name__ == '__main__':
    main()
