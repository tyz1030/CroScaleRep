# Once a batch of random samples are generated over a area on the map,
# we would like to check how these samples are distributed.
# This class help do the plotting

import cv2 as cv
import numpy as np


class SamplePlotter():
    def __init__(self, size = None) -> None:
        self.img_plotted = None
        self.rectangles = []
        self.current_rctg = None
        self.rctg_size = size
        self.__thickness = 1
        self.__font_size = 1
        self.__color = (0, 255, 0)
        return

    def set_line_thickness(self, thickness):
        self.__thickness = thickness
        return
    
    def set_font_size(self, font_size):
        self.__font_size = font_size
        return

    def set_color(self, color):
        self.__color = color
        return
    
    def reset_rctgs(self):
        self.rectangles = []
        return

    def set_rctg_size(self, size):
        self.rctg_size = np.array(size)
        return

    def set_background(self, image):
        self.img_plotted = cv.cvtColor(image.astype(np.float32), cv.COLOR_GRAY2RGB)
        return

    def add_sample(self, pix_idx, rot_angle):
        self.rectangles.append((pix_idx, rot_angle))
        self.current_rctg = (pix_idx, rot_angle)
        return

    def draw_rectangles(self):
        img = np.copy(self.img_plotted)
        for idx, (pix_idx, rot_angle) in enumerate(self.rectangles):
            centre = np.array((pix_idx[1], pix_idx[0]))
            img = self.__draw_a_rectangle(img, centre, rot_angle, self.rctg_size, idx = idx)
        return img

    def draw_current_rectangle(self):
        img = np.copy(self.img_plotted)
        centre = np.array((self.current_rctg[0][1], self.current_rctg[0][0]))
        rot_angle = self.current_rctg[1]
        img = self.__draw_a_rectangle(img, centre, rot_angle, self.rctg_size)
        return img

    def draw_circles(self):
        img = np.copy(self.img_plotted)
        for idx, (pix_idx, rot_angle) in enumerate(self.rectangles):
            centre = (int(pix_idx[1]), int(pix_idx[0]))
            img = self.__draw_a_circle(img, centre)
        return img
    
    def __draw_a_circle(self, image, centre):
        image = cv.circle(image, centre, 1, self.__color, self.__thickness)
        return image

    def __draw_a_rectangle(self, image, centre, theta, size, idx = None):
        '''
        Draw a rectangle on the image
        Input:
            image: input background image matrix
            centre: centre pixel coordinate to draw the rectangle
            theta: rotation angle of the rectangle
            width: width of rectangle
            height: height of rectangle

        Return:
            image with a rectangle
        '''
        theta = np.radians(theta)
        c, s = np.cos(theta), np.sin(theta)
        height = size[0]
        width = size[1]
        R_mat = np.matrix('{} {}; {} {}'.format(c, -s, s, c)) # what's this fancy syntax
        pts = []
        pts.append([+ width / 2, - height / 2])
        pts.append([+ width / 2, + height / 2])
        pts.append([- width / 2, + height / 2])
        pts.append([- width / 2, - height / 2])
        p_rotated = []
        for pt in pts:
            p_rotated.append(np.dot(pt, R_mat.transpose()) + centre)
        p_rotated2 = p_rotated[1:] + [p_rotated[0]] # put first element to last
        color = (0, 255, 0)
        # Draw 4 lines       
        for p1, p2 in zip(p_rotated, p_rotated2):
            cv.line(image, (int(p1[0, 0]), int(
                p1[0, 1])), (int(p2[0, 0]), int(p2[0, 1])), self.__color, self.__thickness)
        # Draw upright direction
        image = cv.line(image, (int(centre[0]), int(
                centre[1])), (int(p1[0, 0]/2.0 + p2[0, 0]/2.0), int(p1[0, 1]/2.0 + p2[0, 1]/2.0)), self.__color, self.__thickness)
        if idx is not None:
            image =  cv.putText(image, str(idx), (centre[0]+int(height/2), centre[1]-int(width/2)), cv.FONT_HERSHEY_PLAIN, self.__font_size, self.__color, self.__thickness) 
        return image
