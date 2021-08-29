
'''
This class is the Cross-Scope Hierarchical Sampler (CSHSampler).
CSHSampler samples the telescope first, then samples the mircoscope in the range of telescope.
A CSHSampler is initialized with one telescope and one microscope sampler. 
'''

import os
import cv2 as cv
import numpy as np
from .sampler_utils import SamplePlotter
from .sampler_utils import CoordRecorder

class CSHSampler():
    def __init__(self, sampler_tele, sampler_micro, output_path, num_teles = None, num_micros = None) -> None:

        self.num_teles = num_teles
        self.num_micros = num_micros

        self.sampler_tele = sampler_tele
        self.sampler_micro = sampler_micro
        self.sampler_micro.register_tele_callback(self.sampler_tele.get_sample_attribute)
        self.output_path = output_path

        # initialize plotter to visualize sampling
        self.tele_plotter = SamplePlotter(size = self.sampler_tele.sample_size)
        self.micro_plotter = SamplePlotter(size = (16, 16)) # hardcoded size

        # initialize recorder for the pixel coordinate from random sampling
        self.coord_recorder = CoordRecorder()
        pass

    def sample(self):
        # Path Validation
        path = self.output_path
        if not os.path.exists(path):
            os.makedirs(path)
        elif len(os.listdir(path)) != 0:
            print("Cross Scope Sampler: Non-empty directory. Stopped.")
            return

        # Sampling
        self.tele_plotter.set_background(self.sampler_tele.data[:, :, 0])
        self.tele_plotter.set_font_size(5)
        self.tele_plotter.set_line_thickness(5)
        self.tele_plotter.set_color((0, 0, 255))
        self.tele_plotter.reset_rctgs()
        for i in range(self.sampler_tele.data.shape[-1]):
            cv.imwrite(os.path.join(path, "tele_"+str(i)+".png"), self.sampler_tele.data[:,:,i])

        for idx_tele in range(self.num_teles):
            print("Sampling Data " + str(idx_tele+1) + "/" + str(self.num_teles))
            subpath = os.path.join(path, str(idx_tele).zfill(5))            
            os.makedirs(subpath)

            # sample telescope
            filename_tele = "tele"
            fullname_tele = os.path.join(subpath, filename_tele)
            tele_sample, pix_coord_tele, rot_angle_tele = self.sampler_tele.sample(subpath)
            self.tele_plotter.add_sample(pix_coord_tele, rot_angle_tele)
            for i in range(tele_sample.shape[-1]):
                cv.imwrite(fullname_tele + str(i) + ".png", tele_sample[:, :, i])
            self.micro_plotter.set_background(tele_sample[:, :, 0])
            self.micro_plotter.reset_rctgs()
            self.coord_recorder.reset()
            # reset pointers in micro sampler
            self.sampler_micro.pixel_selector.reset_iterator()
            for idx_micro in range(self.num_micros):
                # sample microscope
                filename_micro = "micro" + str(idx_micro).zfill(3) + ".png"
                fullname_micro = os.path.join(subpath, filename_micro)
                micro_sample, pix_coord_micro, rot_angle_micro = self.sampler_micro.sample(subpath, idx_micro)
                cv.imwrite(fullname_micro, micro_sample)
                self.micro_plotter.add_sample(pix_coord_micro, rot_angle_micro-rot_angle_tele)
                self.coord_recorder.add_record(pix_coord_micro)
            microplot_name = "micro_samps.png"
            fullname_microplot = os.path.join(subpath, microplot_name)
            cv.imwrite(fullname_microplot, self.micro_plotter.draw_rectangles())
            filename_single_tele_plot = "tele_samp.png"
            fullname_single_tele_plot = os.path.join(subpath, filename_single_tele_plot)
            cv.imwrite(fullname_single_tele_plot, cv.resize(self.tele_plotter.draw_current_rectangle(), (512, 512)))
            self.coord_recorder.write(subpath)
        fullname_teleplot = os.path.join(path, "tele_samps.png")
        cv.imwrite(fullname_teleplot, self.tele_plotter.draw_circles())
        return