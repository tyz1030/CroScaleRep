# Description:
# author TIANYI ZHANG
# version 0.1 Jan 2021
# Copyright (c) 2021 TIANYI ZHANG, DROP Lab, University of Michigan, Ann Arbor


#!/usr/bin/env python3

import numpy as np
import cv2 as cv


class ImgAugmenter():
    def __init__(self) -> None:
        self.f_rotation = False
        self.f_blur = False

    def augment_img(self, img, rot_center=None):
        aug_img = img
        rot_angle = None
        if rot_center == None:
            h, w = aug_img[0].shape[:2]  # height and width
            rot_center = (w//2, h//2)  # img center in x, y
        if self.f_rotation:
            aug_img, rot_angle = self.rotate(aug_img, rot_center)
        if self.f_blur:
            aug_img = self.blur(aug_img)
        return aug_img, rot_angle

    def augmenter_setup(self, rotation=True, blur=True):
        self.f_rotation = rotation
        self.f_blur = blur
        return True

    def rotate_batch(self, imgs, rot_center):
        rot_imgs = []
        ''' gen rand rotation around center '''
        rand_angle = np.random.randint(360)
        rot_mat = cv.getRotationMatrix2D(rot_center, rand_angle, 1)
        for img in imgs:
            rot_imgs.append(cv.warpAffine(img, rot_mat, img.shape[:2]))

        return rot_imgs, rand_angle

    def rand_rotate(self, img, rot_center = None):
        ''' generate counter-clock-wise random rotation around center 
        Input:
        imgs: an numpy array. have dimension Height*Width*Channels
        rot_center: center of rotation (x, y)   -------->  x
                                                |
                                                |
                                                v y

        Return: 
        rot_imgs: an numpy array. have dimension Height*Width*Channels
        rand_angle: random int in [0, 360).
        '''
        rand_angle = np.random.randint(360)
        if rot_center is None:
            rot_center = (img.shape[0]/2.0, img.shape[1]/2.0)
        rot_mat = cv.getRotationMatrix2D(rot_center, rand_angle, 1)
        rot_imgs = cv.warpAffine(img, rot_mat, img.shape[:2])
        rot_imgs = np.reshape(rot_imgs, img.shape)
        return rot_imgs, rand_angle, rot_mat

    def fixed_rotate(self, img, rot_angle = 0, rot_center = None):
        ''' generate counter-clock-wise fixed rotation around center 
        Input:
        imgs: an numpy array. have dimension Height*Width*Channels
        rot_center: center of rotation (x, y)   -------->  x
                                                |
                                                |
                                                v y

        Return: 
        rot_imgs: an numpy array. have dimension Height*Width*Channels
        rand_angle: random int in [0, 360).
        '''
        if rot_center is None:
            rot_center = (img.shape[0]/2.0, img.shape[1]/2.0)
        rot_mat = cv.getRotationMatrix2D(rot_center, rot_angle, 1)
        rot_imgs = cv.warpAffine(img, rot_mat, img.shape[:2])
        rot_imgs = np.reshape(rot_imgs, img.shape)
        return rot_imgs, rot_angle, rot_mat

    def group_rand_rot_90(self, img):
        rotate_decision = np.random.randint(4, dtype=int)
        cv_rot_code = None
        if rotate_decision == 0:
            return img
        if rotate_decision == 1:
            cv_rot_code = cv.ROTATE_90_COUNTERCLOCKWISE
        elif rotate_decision == 2:
            cv_rot_code = cv.ROTATE_180
        elif rotate_decision == 3:
            cv_rot_code = cv.ROTATE_90_CLOCKWISE
        img = cv.rotate(img, cv_rot_code)
        return img

    def group_rand_flip(self, img):
        flip_decision = np.random.randint(0, high= 2, dtype=int) # either flip vertically or horizontally
        img = cv.flip(img, flip_decision) 
        return img

    def group_ops(self, img):
        img = self.group_rand_flip(img)
        img = self.group_rand_rot_90(img)
        return img


def main():
    img_augmenter = ImgAugmenter()
    img_augmenter.augmenter_setup(rotation=True, blur=True)
    img = cv.imread('testimg512.png')
    img_aug = img_augmenter.augment_img(img)

    cv.imshow('testwin', img_aug)
    cv.waitKey(0)
    return True


if __name__ == '__main__':
    main()
