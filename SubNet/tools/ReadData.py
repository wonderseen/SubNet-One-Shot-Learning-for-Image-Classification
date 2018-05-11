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

import numpy as np
import os
import scipy.misc
import Augement
import sys
reload(sys)
sys.setdefaultencoding('utf8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import random


file_addr = '.'
height = 51
width = 51
max_num = 1
type = 6
channel = 1

class switch(object):
    def __init__(self, value):
        self.value = value
        self.fall = False

    def __iter__(self):
        """Return the match method once, then stop"""
        yield self.match
        raise StopIteration

    def match(self, *args):
        """Indicate whether or not to enter a case suite"""
        if self.fall or not args:
            return True
        elif self.value in args:  # changed for v1.5, see below
            self.fall = True
            return True
        else:
            return False

def get_next_batch(mode, batch_size=32):
    '''
    :param mode: 'train' or 'test'
    :param batch_size:
    :return:
    '''
    batch_image1 = np.zeros([batch_size, height, width, 1])
    batch_image2 = np.zeros([batch_size, height, width, 1])
    batch_label = np.zeros([batch_size, max_num * 2])

    # randomly choose samples
    if mode == 'train' or mode == 'all':
        for i in range(batch_size*2):
            if i % 2 == 0:
                label1, batch_image1[i//2, :] = get_name_and_image(mode=mode,visualization=False)
            elif i % 2 == 1:
                label2, batch_image2[i//2, :] = get_name_and_image(mode=mode,visualization=False)
            # judge two image are of the same image

            if i % 2 == 1:
                if label1 == label2: # if equals
                    batch_label[i//2,:] = [1,0]
                else:
                    batch_label[i//2,:] = [0,1]

    # if test, use trainset to cluster the testset
    if mode == 'test':
        for i in range(batch_size * 2):
            if i % 2 == 0:
                label1, batch_image1[i // 2, :] = get_name_and_image(mode='test')#,imageAllTypes=type-1)
            else:
                label2, batch_image2[i // 2, :] = get_name_and_image(mode='train')
            # judge two image are of the same image

            if i % 2 == 1:
                if label1 == label2:  # if equals
                    batch_label[i // 2, :] = [1, 0]
                else:
                    batch_label[i // 2, :] = [0, 1]
    return batch_image1, batch_image2, batch_label

def name2vec(name):
    vector = name.copy()
    return vector

def get_name_and_image(mode, imageAllTypes=type, visualization=False, type_2=True):
    '''
    :param mode: 'train' or 'test'
    :param imageAllTypes: always equals 2 because only pos-image and neg-image
    :return: name, image
    '''

    # get address
    set = file_addr + '/' + mode + 'set/'
    subject_directory = os.listdir(set)

    # to overcome sample imbalance
    if random.randint(0,1) == 0:
        random_type = random.randint(0, 0)
    else:
        random_type = random.randint(1, imageAllTypes-1)

    sample_directory = os.listdir(set + subject_directory[random_type])
    type_directory = os.listdir(set + subject_directory[random_type])
    random_file_arg = random.randint(0, len(type_directory)-1)
    image_addr = set + subject_directory[random_type] + '/' + sample_directory[random_file_arg]

    # get label
    name = subject_directory[random_type]

    # if only classify 2 situations, just turn type_2=True
    if type_2:
        if name == '0':
            name = '0'
        else:
            name = '1'


    # get image
    image = scipy.misc.imread(image_addr)

    # visualization
    if visualization == True:
        fig = plt.figure('1')
        ax1 = fig.add_subplot(111)
        ax1.imshow(image, cmap=plt.cm.gray)
        plt.show()

    if mode == 'train' or mode == 'all':
        image = Augement.sample_augement(image)

    image = image.reshape([height,width,channel]).astype('float32')
    image = (image-np.mean(image))/(np.max(image)-np.min(image))
    return name, image
