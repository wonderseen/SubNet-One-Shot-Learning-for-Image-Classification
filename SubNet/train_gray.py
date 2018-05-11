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
import SubNet
import Draw_ROC_iteration

if __name__ == '__main__':
    model_addr = './model/classification.ckpt-'
    step = 0
    GRAY_net = Network2.CL_Network(start_step=step)
    GRAY_net.train(continueflag=True, model_addr=model_addr+str(step),debug_labels=False)
    #GRAY_net.test_single_threshold(model_addr=model_addr+str(step), test_num=100, test_step=100, threshold=0.5, debug_labels=False)
    #GRAY_net.get_ROC(model_addr=model_addr+str(step), test_num=1000, saveResult=True, saveTXTName=str(step)+'.txt')
    #GRAY_net.test(model_addr=model_addr+str(step), test_num=100, debug_labels=False)
    #GRAY_net.test_distance(model_addr=model_addr+str(step),test_num=100,test_step=50)
    #Draw_ROC_iteration.compuse(txtName = ["1","100","200","300","400","500","600"])

