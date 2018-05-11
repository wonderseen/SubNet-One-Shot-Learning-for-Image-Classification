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

import os
import pickle
import tensorflow as tf
import ReadData
import sys
import numpy as np
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
reload(sys)
sys.setdefaultencoding('utf8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt


class CL_Network(object):
    def __init__(self,sess=None,graph=None,start_step=0):
        # input
        self.height = 51
        self.width = 51
        self.channel = 1

        # label
        self.classification_type = 2 # 2种情况,要么相同[1,0],要么不同[0,1]
        self.max_num = 1 # 一张图中类型可能重复出现次数

        # net parameter
        self.w_alpha = 0.1
        self.b_alpha = 0.1
        self.start_lr = 1e-2

        # 初始学习率 [1000:0.1, 4000:0.01, 1e-5]
        self.final_accuracy = 0.90
        self.trainstep = 50
        self.savestep = 100

        # build off graph
        if graph is None:
            self.graph = tf.Graph()

        with self.graph.as_default():
            if sess is None:
                self.session = tf.Session()
            self.global_step = tf.Variable(start_step, trainable=False,name='global_step')
            self.add_global = None
            self.learning_rate = None
            self.global_loss = None
            self.optimizer = None
            self.input_frame_layer1 = None
            self.input_frame_layer2 = None
            self.groundtruth_label_layer = None
            self.keep_prob = None
            self.accuracy = None
            self.network_initial = False
            self.saver = None
            self.predict_out = None
            self.max_idx_p = None
            self.max_idx_truth = None
            self.groundtruth_label_layer = None
            self.groundtruth_label_layer_inner = None
            self.groundtruth_label_layer_outer = None
            self.loss_2d_l2_inner = None
            self.loss_2d_l2_outer = None
            self.softmax_loss = None
            self.label_l2_loss = None
            self.feature_layer_1 = None
            self.feature_layer_2 = None
            self.sub_2d_feature = None
            self.sub_2d_feature_pca = None
            self.encoder_layer = None

    def init(self, weight_files=None, exclude_var_list=None):
        if exclude_var_list is None:
            exclude_var_list = list()

        if weight_files is None:
            weight_files = ['./weights/rgb.pickle', './weights/finetuned.pickle']

        for file_name in weight_files:
            assert os.path.exists(file_name),"File not found."
            with open(file_name, 'rb') as fi:
                weight_dict = pickle.load(fi)
                weight_dict = {k: v for k, v in weight_dict.items() if not any([x in k for x in exclude_var_list])}
                if len(weight_dict) > 0:
                    init_op, init_feed = tf.contrib.framework.assign_from_values(weight_dict)
                    self.session.run(init_op, init_feed)
                    print('Loaded %d variables from %s' % (len(weight_dict), file_name))

    def inference(self):
        frames_input, groundtruth_label_layer, output_predict, keep_prob, global_loss, optimizer = self.inference_classification()
        return frames_input, groundtruth_label_layer, output_predict, keep_prob, global_loss, optimizer

    def inference_classification(self):
        self.network_initial = True
        with self.graph.as_default():
            # initial input output
            self.input_frame_layer1 = tf.placeholder(tf.float32, [None, self.height, self.width, self.channel],name='INPUT_IMAGE1')
            self.input_frame_layer2 = tf.placeholder(tf.float32, [None, self.height, self.width, self.channel],name='INPUT_IMAGE2')
            self.groundtruth_label_layer = tf.placeholder(tf.float32, [None, self.classification_type * self.max_num], name='GROUNDTRUTH')
            self.groundtruth_label_layer_inner = tf.placeholder(tf.float32, [None, self.classification_type/2 * self.max_num], name='GROUNDTRUTH_INNER')
            self.groundtruth_label_layer_outer = tf.placeholder(tf.float32, [None, self.classification_type/2 * self.max_num], name='GROUNDTRUTH_OUTER')
            self.keep_prob = tf.placeholder(tf.float32)

            ##########################################################################################################
            # image1 特征提取
            # 第1层提取层
            w_c1_1 = tf.Variable(self.w_alpha * tf.random_normal([5, 5, 1, 8]))
            b_c1_1 = tf.Variable(self.b_alpha * tf.random_normal([8]))
            conv1_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.input_frame_layer1, w_c1_1, strides=[1, 1, 1, 1], padding='VALID'), b_c1_1))
            # 49*49

            w_c1_2 = tf.Variable(self.w_alpha * tf.random_normal([3, 3, 8, 16]))
            b_c1_2 = tf.Variable(self.b_alpha * tf.random_normal([16]))
            conv1_2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1_1, w_c1_2, strides=[1, 1, 1, 1], padding='VALID'), b_c1_2))

            # 第2层提取层
            w_c1_3 = tf.Variable(self.w_alpha * tf.random_normal([3, 3, 16, 20]))
            b_c1_3 = tf.Variable(self.b_alpha * tf.random_normal([20]))
            conv1_3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1_2, w_c1_3, strides=[1, 1, 1, 1], padding='VALID'), b_c1_3))
            # 47*47
            w_c1_4 = tf.Variable(self.w_alpha * tf.random_normal([3, 3, 20, 18]))
            b_c1_4 = tf.Variable(self.b_alpha * tf.random_normal([18]))
            conv1_4 = tf.nn.leaky_relu(tf.nn.bias_add(tf.nn.conv2d(conv1_3, w_c1_4, strides=[1, 1, 1, 1], padding='VALID'), b_c1_4))
            # 41*41

            w_c1_5 = tf.Variable(self.w_alpha * tf.random_normal([3, 3, 18, 12]))
            b_c1_5 = tf.Variable(self.b_alpha * tf.random_normal([12]))
            self.feature_layer_1 = tf.nn.tanh(tf.nn.bias_add(tf.nn.conv2d(conv1_4, w_c1_5, strides=[1, 2, 2, 1], padding='VALID'), b_c1_5))
            # 20*20

            ##########################################################################################################
            # image2 特征提取
            w_c2_1 = tf.Variable(self.w_alpha * tf.random_normal([5, 5, 1, 8]))
            b_c2_1 = tf.Variable(self.b_alpha * tf.random_normal([8]))
            conv2_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.input_frame_layer2, w_c2_1, strides=[1, 1, 1, 1], padding='VALID'),b_c2_1))
            # 49*49

            w_c2_2 = tf.Variable(self.w_alpha * tf.random_normal([3, 3, 8, 16]))
            b_c2_2 = tf.Variable(self.b_alpha * tf.random_normal([16]))
            conv2_2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2_1, w_c2_2, strides=[1, 1, 1, 1], padding='VALID'),b_c2_2))

            # 第2层提取层
            w_c2_3 = tf.Variable(self.w_alpha * tf.random_normal([3, 3, 16, 20]))
            b_c2_3 = tf.Variable(self.b_alpha * tf.random_normal([20]))
            conv2_3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2_2, w_c2_3, strides=[1, 1, 1, 1], padding='VALID'),b_c2_3))
            # 47*47
            w_c2_4 = tf.Variable(self.w_alpha * tf.random_normal([3, 3, 20, 18]))
            b_c2_4 = tf.Variable(self.b_alpha * tf.random_normal([18]))
            conv2_4 = tf.nn.leaky_relu(tf.nn.bias_add(tf.nn.conv2d(conv2_3, w_c2_4, strides=[1, 1, 1, 1], padding='VALID'), b_c2_4))
            # 41*41

            w_c2_5 = tf.Variable(self.w_alpha * tf.random_normal([3, 3, 18, 12]))
            b_c2_5 = tf.Variable(self.b_alpha * tf.random_normal([12]))
            self.feature_layer_2 = tf.nn.tanh(tf.nn.bias_add(tf.nn.conv2d(conv2_4, w_c2_5, strides=[1, 2, 2, 1], padding='VALID'), b_c2_5),name='feature_layer_2')

            ##########################################################################################################
            # image1_2 分类网络
            self.sub_2d_feature = tf.subtract(self.feature_layer_1, self.feature_layer_2, 'feature_subtract')
            w_pca = tf.Variable(self.w_alpha * tf.random_normal([20 * 20 * 12, 128]))
            b_pca = tf.Variable(self.b_alpha * tf.random_normal([128]))
            dense = tf.nn.l2_normalize(tf.reshape(self.sub_2d_feature, [-1, w_pca.get_shape().as_list()[0]]),dim=1)# 对W或者向量正则化
            self.sub_2d_feature_pca = tf.nn.leaky_relu(tf.matmul(dense, w_pca) + b_pca)


            sub_feature_bn = self.bn_layer(self.sub_2d_feature_pca, is_training=True)
            # when bn is added, the performance gets better
            # if bn isn't added, the output could be all_zeros or all_ones

            feature_layer_3 = tf.nn.dropout(sub_feature_bn, self.keep_prob)

            ##########################################################################################################
            # 第4层提取层
            w_c3_encoder = tf.Variable(self.w_alpha * tf.random_normal([128, 16]))
            b_c3_endocder = tf.Variable(self.b_alpha * tf.random_normal([16]))
            self.encoder_layer = tf.nn.leaky_relu(tf.add(tf.matmul(feature_layer_3, w_c3_encoder), b_c3_endocder),name='encoder_layer')

            w_out = tf.Variable(self.w_alpha * tf.random_normal([16, self.classification_type * self.max_num]))
            b_out = tf.Variable(self.b_alpha * tf.random_normal([self.classification_type * self.max_num]))
            self.predict_out = tf.matmul(self.encoder_layer, w_out)

            #########################################################################################################
            # learning_rate
            self.learning_rate = tf.train.exponential_decay(learning_rate=self.start_lr, global_step=self.global_step,decay_steps=100, decay_rate=0.8,name='lr_layer')
            self.loss_2d_l2_inner = tf.reduce_mean(tf.square(sub_feature_bn) * self.groundtruth_label_layer_inner, name='loss_2d_l2_inner')
            self.loss_2d_l2_outer = 100./tf.reduce_mean(tf.square(sub_feature_bn) * self.groundtruth_label_layer_outer, name='loss_2d_l2_inner')
            #self.softmax_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.predict_out, labels=self.groundtruth_label_layer, name='softmax_loss')
            self.label_l2_loss = tf.reduce_mean(tf.square(self.predict_out-self.groundtruth_label_layer), name='label_l2_loss')
            self.global_loss = 0.2 * self.loss_2d_l2_inner + \
                               0.5 * self.loss_2d_l2_outer + \
                               0.3 * self.label_l2_loss  # 逻辑回归利用交叉熵去数据均值
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.global_loss)

            # evaulation
            predict = tf.reshape(self.predict_out, [-1, self.classification_type, self.max_num])
            self.max_idx_p = tf.argmax(predict, axis=1)
            max_idx_truth = tf.reshape(self.groundtruth_label_layer, [-1, self.classification_type, self.max_num])
            self.max_idx_truth = tf.argmax(max_idx_truth, axis=1)
            correct_pred = tf.equal(self.max_idx_p, self.max_idx_truth)
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32),name='accuracy')
            self.saver = tf.train.Saver()

        return self.input_frame_layer1, self.groundtruth_label_layer, self.predict_out, self.keep_prob

    def triplet_loss(anchor, positive, negative, alpha):
        """Calculate the triplet loss according to the FaceNet paper

        Args:
          anchor: the embeddings for the anchor images.
          positive: the embeddings for the positive images.
          negative: the embeddings for the negative images.

        Returns:
          the triplet loss according to the FaceNet paper as a float tensor.
        """
        with tf.variable_scope('triplet_loss'):
            pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
            neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

            basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
            loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)

        return loss

    def train(self, continueflag, model_addr, debug_labels=False, visual_loss=False):
        '''
        :param continueflag: True or False
        :param model_addr: Address of the stored model and to store the model
        :return:
        '''

        if self.network_initial == False:
            self.inference_classification()

        lossy = [[],[],[],[]]
        with self.session.as_default():
            with self.graph.as_default():
                self.add_global = self.global_step.assign_add(1)
                if continueflag == True:
                    self.saver.restore(self.session, model_addr)
                    #self.session.run([self.return_global])
                else:
                    self.session.run(tf.global_variables_initializer())

                # training loop
                while True:
                    batch_inputs1, batch_inputs2, batch_labels = ReadData.get_next_batch(mode='train', batch_size=32)
                    batch_labels_inner = batch_labels[:,0]
                    batch_labels_inner = batch_labels_inner.reshape(32,1)
                    batch_labels_outer = batch_labels[:,1]
                    batch_labels_outer = batch_labels_outer.reshape(32,1)
                    step, lr= self.session.run([self.add_global, self.learning_rate])  # 更新学习率

                    _, train_loss, inner, outer = self.session.run([self.optimizer, self.global_loss, self.loss_2d_l2_inner, self.loss_2d_l2_outer],
                                                  feed_dict={
                                                            self.input_frame_layer1: batch_inputs1,
                                                            self.input_frame_layer2: batch_inputs2,
                                                            self.groundtruth_label_layer: batch_labels,
                                                            self.groundtruth_label_layer_inner: batch_labels_inner,
                                                            self.groundtruth_label_layer_outer: batch_labels_outer,
                                                            self.keep_prob: 0.8})

                    if step % self.trainstep == 0:
                        # test loss
                        batch_x_test1, batch_x_test2, batch_labels = ReadData.get_next_batch(mode='test', batch_size=32)
                        batch_labels_inner = batch_labels[:, 0]
                        batch_labels_inner = batch_labels_inner.reshape(32, 1)
                        batch_labels_outer = batch_labels[:, 1]
                        batch_labels_outer = batch_labels_outer.reshape(32, 1)
                        [test_loss, test_acc, max_idx_p, max_idx_truth] = self.session.run(
                                                    [self.global_loss, self.accuracy, self.max_idx_p, self.max_idx_truth],
                                            feed_dict={ self.input_frame_layer1: batch_x_test1,
                                                        self.input_frame_layer2: batch_x_test2,
                                                        self.groundtruth_label_layer: batch_labels,
                                                        self.groundtruth_label_layer_inner: batch_labels_inner,
                                                        self.groundtruth_label_layer_outer: batch_labels_outer,
                                                        self.keep_prob: 1.
                                                      })

                        # train loss
                        batch_x_test1, batch_x_test2, batch_labels = ReadData.get_next_batch(mode='train', batch_size=32)
                        batch_labels_inner = batch_labels[:, 0]
                        batch_labels_inner = batch_labels_inner.reshape(32, 1)
                        batch_labels_outer = batch_labels[:, 1]
                        batch_labels_outer = batch_labels_outer.reshape(32, 1)
                        [train_loss, train_acc] = self.session.run([self.global_loss, self.accuracy],
                                                                    feed_dict={self.input_frame_layer1: batch_x_test1,
                                                                               self.input_frame_layer2: batch_x_test2,
                                                                               self.groundtruth_label_layer: batch_labels,
                                                                               self.groundtruth_label_layer_inner: batch_labels_inner,
                                                                               self.groundtruth_label_layer_outer: batch_labels_outer,
                                                                               self.keep_prob: 1.})

                        print 'Step result: ', step, 'test_accuracy:', test_acc, 'train_accuracy:', train_acc
                        if debug_labels == True:
                            print 'max_idx_p: ', max_idx_p.reshape(1,-1)
                            print 'max_idx_truth: ', max_idx_truth.reshape(1,-1)

                        # save model
                        if step % self.savestep == 0:
                            self.saver.save(self.session, "./model/classification.ckpt", global_step=step)
                            tf.train.write_graph(self.graph, "./cloudresult", "nn_model.pbtxt", False)  # as_text=True)

                        if test_acc > self.final_accuracy and train_acc > self.final_accuracy:
                            self.saver.save(self.session, "./model/classification.ckpt", global_step=step)
                            plt.show()
                            break

                        # 显示训练过程的loss变化过程:
                        if visual_loss:
                            lossy[2].append(test_acc)
                            lossy[3].append(train_acc)
                    print 'step= ', step, 'lr= ', lr, 'train_loss= ', train_loss  # ,'test_loss= ', test_loss

                    # 显示训练过程的loss变化过程:
                    if visual_loss:
                        lossy[0].append(step)
                        lossy[1].append(train_loss)
                        plt.clf()
                        plt.xlabel('/Step', fontsize=15)
                        plt.ylabel('/LOSS', fontsize=15)
                        plt.title('Classification Test Loss', fontsize=18)
                        plt.ylim(0, 1.0)
                        plt.grid(True, linestyle="-.", color="black", linewidth="1")
                        plt.plot(lossy[0], lossy[1], color='blue')
                        plt.pause(0.01)

    def test_discard_all_negative(self, model_addr,test_num=20,debug_labels=False):
        '''
        :param continueflag: True or False
        :param model_addr: Address of the stored model and to store the model
        :return:
        '''

        if self.network_initial == False:
            self.inference_classification()

        accuracy = [[],[],[]]
        with self.session.as_default():
            with self.graph.as_default():
                self.add_global = self.global_step.assign_add(1)
                self.saver.restore(self.session, model_addr)

                # test loop
                step = 0
                while step < test_num:
                    # test loss
                    step += 1
                    batch_x_test1, batch_x_test2, groundtruth_label = ReadData.get_next_batch(mode='test', batch_size=100)
                    [predict_out, test_acc, max_idx_truth, max_idx_p] = self.session.run([self.predict_out, self.accuracy, self.max_idx_truth, self.max_idx_p],
                                        feed_dict={self.input_frame_layer1: batch_x_test1,
                                                   self.input_frame_layer2: batch_x_test2,
                                                   self.groundtruth_label_layer: groundtruth_label,
                                                   self.keep_prob: 1.})
                    # 手动算正确率
                    # 去除哪些判断过程模棱两可的 ,比如 predict_out = [0.49,0.51]
                    test_acc_threshold = self.calculate_precious_discard_all_negative(predict_out, max_idx_truth, max_idx_p)

                    print 'step =', step, 'test_accuracy:', test_acc, 'threshold_accuracy:', test_acc_threshold
                    if debug_labels:
                        print 'max_idx_p: ', max_idx_p.reshape(1,-1)
                        print 'max_idx_truth: ', max_idx_truth.reshape(1,-1)

                    # 显示训练过程的loss_变化过程:.'
                    accuracy[2].append(test_acc)
                    accuracy[1].append(test_acc_threshold)
                    accuracy[0].append(step)

                plt.clf()
                plt.xlabel('/Step', fontsize=15)
                plt.ylabel('/Accuracy', fontsize=15)
                plt.title('Classification Test Accuracy', fontsize=18)
                plt.ylim(0, 1.05)
                #plt.grid(True, linestyle="-.", color="black", linewidth="1")
                plt.plot(accuracy[0], accuracy[1], color='blue')
                plt.plot(accuracy[0], accuracy[2], color='red')

                mean_acc_threshold = np.mean(accuracy[1])
                mean_acc = np.mean(accuracy[2])
                mean = []
                mean_at = []
                for i in range(0,test_num):
                    mean.extend([mean_acc])
                    mean_at.extend([mean_acc_threshold])
                print 'step =', step, 'mean_accuracy = ', mean[0], 'mean_accuracy_threshold = ', mean_at[0]

                plt.plot(accuracy[0], mean_at, 'b--')
                plt.plot(accuracy[0], mean, 'r--')
                plt.legend(('Test without threshold', 'Test with threshold', 'Mean with Op = '+str(round(mean_at[0], 4)), 'Mean without Op = ' + str(round(mean[0],4)),),loc='center left', bbox_to_anchor=(0.5, 0.25), ncol=1)
                plt.show()

    def test_discard_all_negative(self, model_addr,test_num=20, debug_labels=False):
        '''
        :param continueflag: True or False
        :param model_addr: Address of the stored model and to store the model
        :return:
        '''

        if self.network_initial == False:
            self.inference_classification()

        accuracy = [[],[],[]]
        with self.session.as_default():
            with self.graph.as_default():
                self.add_global = self.global_step.assign_add(1)
                self.saver.restore(self.session, model_addr)

                # test loop
                step = 0
                while step < test_num:
                    # test loss
                    step += 1
                    batch_x_test1, batch_x_test2, groundtruth_label = ReadData.get_next_batch(mode='test', batch_size=100)
                    [predict_out, test_acc, max_idx_truth, max_idx_p] = self.session.run([self.predict_out, self.accuracy, self.max_idx_truth, self.max_idx_p],
                                        feed_dict={self.input_frame_layer1: batch_x_test1,
                                                   self.input_frame_layer2: batch_x_test2,
                                                   self.groundtruth_label_layer: groundtruth_label,
                                                   self.keep_prob: 1.})
                    # 手动算正确率
                    # 去除哪些判断过程模棱两可的 ,比如 predict_out = [0.49,0.51]
                    test_acc_threshold = self.calculate_precious_discard_all_negative(predict_out, max_idx_truth, max_idx_p)

                    print 'step =', step, 'test_accuracy:', test_acc, 'threshold_accuracy:', test_acc_threshold

                    if debug_labels:
                        print 'max_idx_p: ', max_idx_p.reshape(1,-1)
                        print 'max_idx_truth: ', max_idx_truth.reshape(1,-1)

                    # 显示训练过程的loss_变化过程:.'
                    accuracy[2].append(test_acc)
                    accuracy[1].append(test_acc_threshold)
                    accuracy[0].append(step)

                plt.clf()
                plt.xlabel('/Step', fontsize=15)
                plt.ylabel('/Accuracy', fontsize=15)
                plt.title('Classification Test Accuracy', fontsize=18)
                plt.ylim(0, 1.05)
                #plt.grid(True, linestyle="-.", color="black", linewidth="1")
                plt.plot(accuracy[0], accuracy[1], color='blue')
                plt.plot(accuracy[0], accuracy[2], color='red')

                mean_acc_threshold = np.mean(accuracy[1])
                mean_acc = np.mean(accuracy[2])
                mean = []
                mean_at = []
                for i in range(0,test_num):
                    mean.extend([mean_acc])
                    mean_at.extend([mean_acc_threshold])
                print 'step =', step, 'mean_accuracy = ', mean[0], 'mean_accuracy_threshold = ', mean_at[0]

                plt.plot(accuracy[0], mean_at, 'b--')
                plt.plot(accuracy[0], mean, 'r--')
                plt.legend(('Test without threshold', 'Test with threshold', 'Mean with Op = '+str(round(mean_at[0], 4)), 'Mean without Op = ' + str(round(mean[0],4)),),loc='center left', bbox_to_anchor=(0.5, 0.25), ncol=1)
                plt.show()

    def test_single_threshold(self, model_addr, test_num=20, test_step=100, threshold=0., debug_labels=False):
        '''
        :param continueflag: True or False
        :param model_addr: Address of the stored model and to store the model
        :return:
        '''

        if self.network_initial == False:
            self.inference_classification()

        accuracy = [[],[],[]]
        abandon_rates = []
        with self.session.as_default():
            with self.graph.as_default():
                self.add_global = self.global_step.assign_add(1)
                self.saver.restore(self.session, model_addr)

                # test loop
                step = 0
                while step < test_step:
                    # test loss
                    step += 1
                    batch_x_test1, batch_x_test2, groundtruth_label = ReadData.get_next_batch(mode='test', batch_size=test_num)
                    [predict_out, test_acc, max_idx_truth, max_idx_p] = self.session.run([self.predict_out, self.accuracy, self.max_idx_truth, self.max_idx_p],
                                        feed_dict={self.input_frame_layer1: batch_x_test1,
                                                   self.input_frame_layer2: batch_x_test2,
                                                   self.groundtruth_label_layer: groundtruth_label,
                                                   self.keep_prob: 1.})

                    # 手动算正确率
                    # 去除哪些判断过程模棱两可的 ,比如 predict_out = [0.49,0.51]
                    test_acc_threshold, abandon_rate = calculate_precious_single_threshold(predict_out, max_idx_truth, max_idx_p,threshold=threshold)

                    print 'step =', step, 'test_accuracy:', test_acc, 'threshold_accuracy:', test_acc_threshold

                    if debug_labels:
                        print 'max_idx_p: ', max_idx_p.reshape(1,-1)
                        print 'max_idx_truth: ', max_idx_truth.reshape(1,-1)

                    # 显示训练过程的loss_变化过程:.'
                    accuracy[2].append(test_acc)
                    accuracy[1].append(test_acc_threshold)
                    accuracy[0].append(step)
                    # 丢弃的pair数量
                    abandon_rates.append(abandon_rate)

                plt.clf()
                plt.xlabel('/Step', fontsize=15)
                plt.ylabel('/Accuracy', fontsize=15)
                plt.title('Classification Test Accuracy', fontsize=18)
                plt.ylim(0, 1.05)
                #plt.grid(True, linestyle="-.", color="black", linewidth="1")
                plt.plot(accuracy[0], accuracy[1], color='blue')
                plt.plot(accuracy[0], accuracy[2], color='red')

                mean_acc_threshold = np.mean(accuracy[1])
                mean_acc = np.mean(accuracy[2])
                mean = []
                mean_at = []
                for i in range(0,test_num):
                    mean.extend([mean_acc])
                    mean_at.extend([mean_acc_threshold])
                print 'step =', step, 'mean_accuracy = ', mean[0], 'mean_accuracy_threshold = ', mean_at[0]
                print 'mean_abandon_rates =', np.mean(abandon_rates)

                plt.plot(accuracy[0], mean_at, 'b--')
                plt.plot(accuracy[0], mean, 'r--')
                plt.legend(('Test without TO', 'Test with TO', 'Mean with TO = '+str(round(mean_at[0], 4)), 'Mean without TO = ' + str(round(mean[0],4)),),loc='center left', bbox_to_anchor=(0.5, 0.25), ncol=1)
                plt.show()

    def get_ROC(self, model_addr, test_num=20, saveResult=False, saveTXTName='default.txt'):
        '''
        :param continueflag: True or False
        :param model_addr: Address of the stored model and to store the model
        :return:
        '''

        if self.network_initial == False:
            self.inference_classification()

        accuracy = [[],[],[]]
        abandon_rates = []
        with self.session.as_default():
            with self.graph.as_default():
                self.add_global = self.global_step.assign_add(1)
                self.saver.restore(self.session, model_addr)

                # test loop
                step = 0
                # test loss
                step += 1
                batch_x_test1, batch_x_test2, groundtruth_label = ReadData.get_next_batch(mode='test', batch_size=test_num)
                [predict_out, test_acc, max_idx_truth, max_idx_p] = self.session.run([self.predict_out, self.accuracy, self.max_idx_truth, self.max_idx_p],
                                    feed_dict={self.input_frame_layer1: batch_x_test1,
                                               self.input_frame_layer2: batch_x_test2,
                                               self.groundtruth_label_layer: groundtruth_label,
                                               self.keep_prob: 1.})
                # 画ROC
                batch_labels_outer = groundtruth_label[:, 1]
                batch_labels_outer = batch_labels_outer.reshape(test_num, 1)
                self.ROC_diagram(predict_out, batch_labels_outer, saveResult=saveResult, saveTXTName=saveTXTName)


    def test(self, model_addr, test_num=20, debug_labels=False):
        '''
        :param continueflag: True or False
        :param model_addr: Address of the stored model and to store the model
        :return:
        '''

        if self.network_initial == False:
            self.inference_classification()

        accuracy = [[], []]
        with self.session.as_default():
            with self.graph.as_default():
                self.add_global = self.global_step.assign_add(1)
                self.saver.restore(self.session, model_addr)

                # test loop
                step = 0
                while step < test_num:
                    # test loss
                    step += 1
                    batch_x_test1, batch_x_test2, groundtruth_label = ReadData.get_next_batch(mode='test',batch_size=100)
                    [predict_out, test_acc, max_idx_truth, max_idx_p] = self.session.run(
                        [self.predict_out, self.accuracy, self.max_idx_truth, self.max_idx_p],
                        feed_dict={self.input_frame_layer1: batch_x_test1,
                                   self.input_frame_layer2: batch_x_test2,
                                   self.groundtruth_label_layer: groundtruth_label,
                                   self.keep_prob: 1.})
                    print 'step =', step, 'test_accuracy:', test_acc

                    if debug_labels:
                        print 'max_idx_p: ', max_idx_p.reshape(1,-1)
                        print 'max_idx_truth: ', max_idx_truth.reshape(1,-1)

                    # 显示训练过程的loss_变化过程:.'
                    accuracy[1].append(test_acc)
                    accuracy[0].append(step)

                plt.clf()
                plt.xlabel('/Step', fontsize=15)
                plt.ylabel('/Accuracy', fontsize=15)
                plt.title('Classification Test Accuracy', fontsize=18)
                plt.ylim(0, 1.05)
                # plt.grid(True, linestyle="-.", color="black", linewidth="1")
                plt.plot(accuracy[0], accuracy[1], color='blue')

                mean_acc = np.mean(accuracy[1])
                mean = []
                for i in range(0, test_num):
                    mean.extend([mean_acc])
                print 'step =', step, 'mean_accuracy = ', mean[0]
                plt.plot(accuracy[0], mean, 'r--')
                plt.legend(('Test Accuracy', 'Mean Accuracy = ' + str(mean[0])), loc='center left',
                           bbox_to_anchor=(0.5, 0.25), ncol=1)
                plt.show()

    def test_distance(self, model_addr=None, test_num=20, test_step=50, debug_labels=False):
        '''
        To evaluate the distance from embeddings of two samples
        :param model_addr: the address of model
        :param test_num: number of sample for every testing
        :param debug_labels: print out groundtruth or not
        :return:
        '''

        if self.network_initial == False:
            self.inference_classification()

        distance = [[],[],[]]
        with self.session.as_default():
            with self.graph.as_default():
                self.add_global = self.global_step.assign_add(1)
                self.saver.restore(self.session, model_addr)

                # test loop
                step = 0
                while step < test_step:
                    print 'step =', step
                    # test loss
                    step += 1
                    batch_x_test1, batch_x_test2, batch_labels = ReadData.get_next_batch(mode='test', batch_size=test_num)
                    batch_labels_inner = batch_labels[:, 0]
                    batch_labels_inner = batch_labels_inner.reshape(test_num, 1)
                    batch_labels_outer = batch_labels[:, 1]
                    batch_labels_outer = batch_labels_outer.reshape(test_num, 1)

                    [sub_2d_feature] = self.session.run([self.encoder_layer],
                                        feed_dict={self.input_frame_layer1: batch_x_test1,
                                                   self.input_frame_layer2: batch_x_test2,
                                                   self.keep_prob: 1.})

                    sub_2d_feature_inner = np.matmul(np.sum(np.square(sub_2d_feature),-1),np.transpose(batch_labels_inner.reshape(-1)))/ np.sum(batch_labels_inner)
                    sub_2d_feature_outer = np.matmul(np.sum(np.square(sub_2d_feature),-1),np.transpose(batch_labels_outer.reshape(-1)))/ np.sum(batch_labels_outer)

                    # 显示训练过程的loss_变化过程:.'
                    distance[2].append(sub_2d_feature_inner)
                    distance[1].append(sub_2d_feature_outer)
                    distance[0].append(step)

                plt.clf()
                plt.xlabel('/Step', fontsize=15)
                plt.ylabel('/Accuracy', fontsize=15)
                plt.title("Distances of Pairs' Feature-Vectors", fontsize=18)
                #plt.grid(True, linestyle="-.", color="black", linewidth="1")
                plt.plot(distance[0], distance[1], color='blue')
                plt.plot(distance[0], distance[2], color='red')

                mean_acc_threshold = np.mean(distance[1])
                mean_acc = np.mean(distance[2])
                mean_in = []
                mean_ot = []
                for i in range(0,test_step):
                    mean_in.extend([mean_acc])
                    mean_ot.extend([mean_acc_threshold])
                print 'step =', step, 'mean_inner = ', mean_in[0], 'mean_outer = ', mean_ot[0]

                plt.plot(distance[0], mean_ot, 'b--')
                plt.plot(distance[0], mean_in, 'r--')
                plt.ylim(0, 650)

                plt.legend(('Between-Class Distance',
                            'Within-Class Distance',
                            'Mean of Between-Class = '+str(round(mean_ot[0], 3)),
                            'Mean of Within-Class = ' + str(round(mean_in[0], 3)),),
                           loc='center left', ncol=1, bbox_to_anchor=(0.35, 0.35))
                plt.show()


    # 实现Batch Normalization
    def bn_layer(self, x, is_training,name='BatchNorm',moving_decay=0.9, eps=1e-5):
        # 获取输入维度并判断是否匹配卷积层(4)或者全连接层(2)
        shape = x.shape
        assert len(shape) in [2,4]

        param_shape = shape[-1]
        with tf.variable_scope(name):
            # 声明BN中唯一需要学习的两个参数，y=gamma*x+beta
            gamma = tf.get_variable('gamma',param_shape,initializer=tf.constant_initializer(1))
            beta  = tf.get_variable('beta', param_shape,initializer=tf.constant_initializer(0))

            # 计算当前整个batch的均值与方差
            axes = list(range(len(shape)-1))
            batch_mean, batch_var = tf.nn.moments(x,axes,name='moments')

            # 采用滑动平均更新均值与方差
            ema = tf.train.ExponentialMovingAverage(moving_decay)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean,batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            # 训练时，更新均值与方差，测试时使用之前最后一次保存的均值与方差
            mean, var = tf.cond(tf.equal(is_training,True),mean_var_with_update,
                    lambda:(ema.average(batch_mean),ema.average(batch_var)))

            # 最后执行batch normalization
            return tf.nn.batch_normalization(x,mean,var,beta,gamma,eps)

    # calculate_ROC
    def ROC_diagram(self, predict_out, groundtruth, saveResult=False, saveTXTName='default.txt'):
        '''
        How we classify the pair by 'predict_out'
        If the  prediction of the ith pair ('predict_out[i]') satisfies the inequality:

            predict_out[i, 0] - predict_out[i, 1] > threshold

        we judge the two images of ith pair are of the same class. Otherwise, we judge the two images of the ith pair are of different classes

        :param predict_out:
        :param groundtruth: if the first dim equals 1, the pair is of same class
        :return:
        '''

        tp_fn = len(groundtruth)-np.sum(groundtruth) # groundtruth positive
        fp_tn = np.sum(groundtruth) # groundtruth negative

        TP_rate = []
        FP_rate = []

        len_predict = len(predict_out)

        # 按照正例概率强到弱从大到小排列
        # https://blog.csdn.net/qq_25964837/article/details/79047948
        predict_out = predict_out[:,0] - predict_out[:,1]
        sort_arg = np.argsort(-predict_out) # 降序排列
        groundtruth = groundtruth.reshape(len_predict)
        groundtruth = groundtruth[sort_arg]
        predict_out = np.sort(predict_out)

        for j in range(0,21):
            threshold_arg = j * len_predict/20.
            tp = 0.
            fp = 0.
            tn = 0.
            fn = 0.
            for i in range(len_predict): # 遍历
                predict = 1.
                if i < threshold_arg: # 前切片
                    predict = 0.
                if predict == 0.: # 如果预测为正例
                    if predict == groundtruth[i]:
                        tp += 1.
                    else:
                        fp += 1.
                elif predict == 1.: # 如果预测为负例
                    if predict == groundtruth[i]:
                        tn += 1.
                    else:
                        fn += 1.
            # TPR = TP/(TP+FN)
            # FPR = FP/(FP+TN)
            TP_rate.append(tp / tp_fn)
            FP_rate.append(fp / fp_tn)

        # 显示
        plt.clf()
        plt.xlabel('/TPR', fontsize=15)
        plt.ylabel('/FPR', fontsize=15)
        plt.title('Receiver Operating Characteristic Curve', fontsize=18)
        plt.ylim(0, 1.01)
        plt.xlim(0, 1.0)
        #plt.grid(True, linestyle="-.", color="black", linewidth="1")
        plt.plot(FP_rate, TP_rate, color='blue')
        plt.plot([0,1], [0,1], 'r--')
        plt.show(0.01)

        if saveResult:
            print 'save the result'
            f = file(saveTXTName, "a")
            for i in range(0, 21):
                new_context = str(FP_rate[i])+' '
                f.write(new_context)
                new_context = str(TP_rate[i]) + '\n'
                f.write(new_context)
            f.close()
            sys.exit()

        return TP_rate, FP_rate

    def calculate_AUC(self, ROC):
        pass

    # calsulate the preciousness
    def calculate_precious_discard_all_negative(self, predict_out, max_idx_truth, max_idx_p):
        tp = 0.
        tp_fn = 0.
        for i in range(len(predict_out)):
            if predict_out[i, 0] < 0 and predict_out[i, 1] < 0:
                pass
            else:
                if max_idx_truth[i] == max_idx_p[i]:
                    tp += 1.
                tp_fn += 1.
        print '被剔除了%d对pair'%(len(predict_out)-tp_fn)
        return tp/tp_fn


# calsulate the preciousness
def calculate_precious_single_threshold(predict_out, max_idx_truth, max_idx_p, threshold):
    tp = 0.
    tp_fn = 0.
    abandon_num = 0.
    for i in range(len(predict_out)):
        if abs(predict_out[i, 0] - predict_out[i, 1]) < threshold:
            abandon_num += 1.
        else:
            if max_idx_truth[i] == max_idx_p[i]:
                tp += 1.
            tp_fn += 1.
    print '被剔除了%d对pair'%(len(predict_out)-tp_fn)
    return tp/tp_fn, abandon_num/len(predict_out)
