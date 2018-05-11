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

import sys
import numpy as np
reload(sys)
sys.setdefaultencoding('utf8')
import matplotlib.pyplot as plt

#https://www.cnblogs.com/darkknightzh/p/6117528.html
def compuse(txtName):
    '''
    The function is to read '.txt' files for drawing ROC chart from different model-test-result
    :param txtName: Form like ["1","150","200","800","1100"]
    :return: Nothing
    '''
    colormaps = ['#4169E1', '#FAA460', '#CD853F', '#6A5ACD', '#2E8B57', '#40E0D0', '#D2B48C', '#F5DEB3', '#B0E0E6']
    plt.clf()
    plt.xlabel('/TPR', fontsize=15)
    plt.ylabel('/FPR', fontsize=15)
    plt.title('Receiver Operating Characteristic Curve', fontsize=18)

    # 1. 读取所有文本的ROC数据
    for i in range(len(txtName)):
        data = np.loadtxt(txtName[i] + '.txt') #  ["1.txt","150.txt","200.txt","800.txt","1100.txt"]
        FP_rate = data[:,0]
        TP_rate = data[:,1]
        plt.plot(FP_rate, TP_rate, color=colormaps[i])

    # 2. 其他绘图限制
    # plt.grid(True, linestyle="-.", color="black", linewidth="1")
    plt.ylim(0, 1.01)
    plt.xlim(0, 1.0)

    # 3. 图像注释
    notes = []
    for i in range(len(txtName)):
        notes.append('iteration-'+ txtName[i]) # notes = ['iteration 1', 'iteration 150', 'iteration 200','iteration 800', 'iteration 1100']
    plt.legend(notes, loc='center left', bbox_to_anchor=(0.6, 0.25), ncol=1)

    # 4. 参考线
    plt.plot([0, 1], [0, 1], 'r--')
    plt.show()