# -*- coding: utf-8 -*-
# Author:
#       WonderSeen, Xiamen University
#
# Up-to-Date:
#       2018.5.10
#
# Github:
#       https://github.com/wonderseen/
#
# Description:
#       I make build this model (SubNet) for one-shot learning in multi-type-image-classification.
#
# Dataset:
#       Not Provided
#
##############################################################################################
##############################################################################################
##                                                                                          ##
##  ##         ##   #####   ###     # #####   ###### ######   #####  ##### ##### ###     #  ##
##  ##    #    ##  #######  # ##    # #   ##  #      ##   ## ##      #     #     # ##    #  ##
##  ##   ###   ## ###   ### #  ##   # #    ## #      ##   ## ##      #     #     #  ##   #  ##
##  ##  ## ##  ## ##     ## #   ##  # #    ## ###### ######   #####  ##### ##### #   ##  #  ##
##  ## ##   ## ## ###   ### #    ## # #    ## #      ## ##        ## #     #     #    ## #  ##
##   ####    ####  #######  #     ### #   ##  #      ##  ##       ## #     #     #     ###  ##
##    ##      ##    #####   #      ## #####   ###### ##   ##  #####  ##### ##### #      ##  ##
##                                                                                          ##
##############################################################################################
##############################################################################################

import random
import numpy as np
from scipy import ndimage
from scipy import misc

IMAGE_SIZE = 51

gamma = [0.1, 0.15, 0.23, 0.36, 0.58, 0.66, 0.88, 0.92]
def sample_augement(sample_img):
    '''sample_img = fill_sample(sample_img)'''

    RGBCROP_SIZE = sample_img.shape[0]
    if random.randint(0, 100) > 50:
        noise_mode = random.randint(0, 6)
        if noise_mode == 0: 
            for i in range(0, random.randint(1, 10)):
                aa = random.randint(0, RGBCROP_SIZE - 1)
                bb = random.randint(0, RGBCROP_SIZE - 1)
                scale1 = random.uniform(0.5, 1.2)
                sample_img[aa, bb] *= scale1

        elif noise_mode == 1:
            for i in range(0, random.randint(1, 5)):
                aa = random.randint(0, RGBCROP_SIZE - 1)
                bb = random.randint(0, RGBCROP_SIZE - 1)
                scale1 = random.randint(-5, 5)
                sample_img[aa, bb] += scale1
                sample_img[aa, bb] = min(255, sample_img[aa, bb])

        elif noise_mode == 2:
            scale1 = random.uniform(0.7, 0.9)
            for i in range(0, RGBCROP_SIZE):
                for j in range(0, RGBCROP_SIZE):
                    sample_img[i, j] *= scale1

        elif noise_mode == 3:
            sigma = random.randint(0,5)
            sample_img = ndimage.gaussian_filter(sample_img,sigma)

        elif noise_mode == 4:
            import cv2
            sample_img = cv2.flip(sample_img, 1)

        elif noise_mode == 5:
            sample_img = sample_img * random.uniform(0.9, 0.998)

        elif noise_mode == 6:
            sample_img = random_rotate_image(sample_img)

    return sample_img


def random_rotate_image(image):
    angle = np.random.uniform(low=-10.0, high=10.0)
    return misc.imrotate(image, angle, 'bicubic')

def fill_sample(sample_img):
    if IMAGE_SIZE > sample_img.shape[0]:
        if sample_img.shape[0] < IMAGE_SIZE:
            aa = (IMAGE_SIZE - sample_img.shape[0]) // 2
            bb = IMAGE_SIZE - sample_img.shape[0] - aa
            sample_img = np.row_stack((np.zeros((int(aa), int(sample_img.shape[1]), 3)), sample_img))
            sample_img = np.row_stack((sample_img, np.zeros((int(bb), int(sample_img.shape[1]), 3))))
    if sample_img.shape[1] < IMAGE_SIZE:
        aa = (IMAGE_SIZE - sample_img.shape[1]) // 2
        bb = IMAGE_SIZE - sample_img.shape[1] - aa
        sample_img = np.column_stack( (np.zeros((int(512), int(aa), 3)), sample_img))
        sample_img = np.column_stack((sample_img, np.zeros((int(512), int(bb), 3)) ))

    return sample_img
