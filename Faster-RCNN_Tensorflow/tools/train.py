# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import tensorflow.contrib.slim as slim
import os, sys
import numpy as np
import time
sys.path.append("../")
from libs.configs import cfgs
from libs.networks import build_whole_network
from data.io.read_tfrecord import next_batch
from libs.box_utils import show_box_in_tensor
from help_utils import tools


os.environ["CUDA_VISIBLE_DEVICES"] = cfgs.GPU_GROUP


# 网友：As recommended in this github repo: https://github.com/yangxue0827/FPN_Tensorflow,
# I'm wondering is your implementation for FPN?
# 作者：We are trying, but the FPN currently implemented is 10% lower than the Faster-RCNN. @MrWanter
def train():

    # Step 1:
    # clw note：传递网络名称如resnet_v1，是否训练is_training，以及每个位置含有anchor box的个数，
    #           构建基本的网络
    faster_rcnn = build_whole_network.DetectionNetwork(base_network_name=cfgs.NET_NAME, is_training=True)
    with tf.name_scope('get_batch'):  # clw note：tf.name_scope 主要结合 tf.Variable() 来使用，方便参数命名管理。
        # clw note：从文件队列、内存队列中读取、组合，得到该batch的内容
        #           主要包括每个批次（目前仅支持批次数目即batch_size=1，也就是这里每次只读出1张图片）
        #           对应的变量包括：图片名称、图片矩阵、ground truth坐标及对应的label，图片中包含的目标数
        #           这些变量的组成结构均为 [批次数目，相应批次中每一幅图片的相关信息]
        img_name_batch, img_batch, gtboxes_and_label_batch, num_objects_batch = \
            next_batch(dataset_name=cfgs.DATASET_NAME,  # 'pascal', 'coco'
                       batch_size=cfgs.BATCH_SIZE,
                       shortside_len=cfgs.IMG_SHORT_SIDE_LEN,
                       is_training=True)
        gtboxes_and_label = tf.reshape(gtboxes_and_label_batch, [-1, 5])
        # clw note：样本个数m不知道，但是对单个样本都有gtboxes的4个坐标，加上1个label共5个值；使用-1来自动计算样本个数

    biases_regularizer = tf.no_regularizer
    weights_regularizer = tf.contrib.layers.l2_regularizer(cfgs.WEIGHT_DECAY)


    # Step 2：
    # clw note：Faster R-CNN网络的搭建！

    # 先看一下下面这个函数arg_scope的声明
    # @tf_contextlib.contextmanager
    # def arg_scope(list_ops_or_scope, **kwargs): 功能是给list_ops中的内容设置默认值，即list中所有元素都用**kargs的参数设置。
    # 有函数修饰符@tf_contextlib.contextmanager修饰arg_scope函数：@之后一般接一个可调用对象为其执行一系列辅助操作，
    # 我们来看一个demo：
    #########################################
    # import time
    # def my_time(func):
    #     print(time.ctime())
    #     return func()
    #
    # @my_time  # 从这里可以看出@time 等价于 time(xxx()),但是这种写法你得考虑python代码的执行顺序
    # def xxx():
    #     print('Hello world!')
    #
    # 运行结果：
    # Wed Jul 26 23:01:21 2017
    # Hello world!
    ##########################################
    # 在这个例子中，xxx函数实现我们的主要功能，打印Hello world，但我们想给xxx函数添加一些辅助操作，让它同时打印出时间，于是我们用
    # 函数修饰符 @ my_time完成这个目标。整个例子的执行流程为调用my_time可调用对象，它接受xxx函数作为参数，先打印时间，再执行xxx函数
    # 详见：https://www.cnblogs.com/zzy-tf/p/9356883.html

    # 来看另一个demo：
    ##########################################
    # with slim.arg_scope(
    #                 [slim.conv2d, slim.max_pool2d, slim.avg_pool2d],stride = 1, padding = 'VALID'):
    #             net = slim.conv2d(inputs, 32, [3, 3], stride = 2, scope = 'Conv2d_1a_3x3')
    #             net = slim.conv2d(net, 32, [3, 3], scope = 'Conv2d_2a_3x3')
    #             net = slim.conv2d(net, 64, [3, 3], padding = 'SAME', scope = 'Conv2d_2b_3x3')
    # 所以，在使用过程中可以直接slim.conv2d( )等函数设置默认参数。例如在下面的代码中，不做单独声明的情况下，
    # slim.conv2d, slim.max_pool2d, slim.avg_pool2d三个函数默认的步长都设为1，padding模式都是'VALID'的。
    # 当然也可以在调用时进行单独声明，只不过一个一个写很麻烦，不如统一给个默认值。
    # 这种参数设置方式在构建网络模型时，尤其是较深的网络时，可以节省时间。
    with slim.arg_scope([slim.conv2d, slim.conv2d_in_plane, slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected],
                        weights_regularizer=weights_regularizer,
                        biases_regularizer=biases_regularizer,
                        biases_initializer=tf.constant_initializer(0.0)): # list as many types of layers as possible,
                                                                          # even if they are not used now

        # build_whole_detection_network功能：构建整体网络架构，包含backbone，RPN网络，Pooling层，以及后续网络。
        # return：网络的最后的预测框，预测的类别信息，预测的概率，以及整体网络和RPN网络的损失，所有的损失被写入到一个字典中。
        final_bbox, final_scores, final_category, loss_dict = faster_rcnn.build_whole_detection_network(
            input_img_batch=img_batch,
            gtboxes_batch=gtboxes_and_label)

    # ----------------------------------------------------------------------------------------------------build loss
    # weight_decay_loss = tf.add_n(slim.losses.get_regularization_losses())
    # weight_decay_loss = tf.add_n(tf.losses.get_regularization_losses())
    rpn_location_loss = loss_dict['rpn_loc_loss']
    rpn_cls_loss = loss_dict['rpn_cls_loss']
    rpn_total_loss = rpn_location_loss + rpn_cls_loss

    fastrcnn_cls_loss = loss_dict['fastrcnn_cls_loss']
    fastrcnn_loc_loss = loss_dict['fastrcnn_loc_loss']
    fastrcnn_total_loss = fastrcnn_cls_loss + fastrcnn_loc_loss

    # clw note：根据论文的公式，最后将RPN网络的（分类，回归）误差与Fast-RCNN的(分类，回归）误差相加后作为总的误差进行训练即可。
    total_loss = rpn_total_loss + fastrcnn_total_loss
    # ____________________________________________________________________________________________________build loss


    # ---------------------------------------------------------------------------------------------------add summary
    tf.summary.scalar('RPN_LOSS/cls_loss', rpn_cls_loss)
    tf.summary.scalar('RPN_LOSS/location_loss', rpn_location_loss)
    tf.summary.scalar('RPN_LOSS/rpn_total_loss', rpn_total_loss)

    tf.summary.scalar('FAST_LOSS/fastrcnn_cls_loss', fastrcnn_cls_loss)
    tf.summary.scalar('FAST_LOSS/fastrcnn_location_loss', fastrcnn_loc_loss)
    tf.summary.scalar('FAST_LOSS/fastrcnn_total_loss', fastrcnn_total_loss)

    tf.summary.scalar('LOSS/total_loss', total_loss)
    # tf.summary.scalar('LOSS/regular_weights', weight_decay_loss)

    gtboxes_in_img = show_box_in_tensor.draw_boxes_with_categories(img_batch=img_batch,
                                                                   boxes=gtboxes_and_label[:, :-1],
                                                                   labels=gtboxes_and_label[:, -1])
    if cfgs.ADD_BOX_IN_TENSORBOARD:
        detections_in_img = show_box_in_tensor.draw_boxes_with_categories_and_scores(img_batch=img_batch,
                                                                                     boxes=final_bbox,
                                                                                     labels=final_category,
                                                                                     scores=final_scores)
        tf.summary.image('Compare/final_detection', detections_in_img)
    tf.summary.image('Compare/gtboxes', gtboxes_in_img)

    # ___________________________________________________________________________________________________add summary

    global_step = slim.get_or_create_global_step()
    lr = tf.train.piecewise_constant(global_step,
                                     boundaries=[np.int64(cfgs.DECAY_STEP[0]), np.int64(cfgs.DECAY_STEP[1])],
                                     values=[cfgs.LR, cfgs.LR / 10., cfgs.LR / 100.])
    tf.summary.scalar('lr', lr)
    optimizer = tf.train.MomentumOptimizer(lr, momentum=cfgs.MOMENTUM) # clw note：选择优化器，可以尝试其他选择，
    # 也可以尝试tf.train.AdamOptimizer(1e-4).minimize(total_loss)

    # ---------------------------------------------------------------------------------------------compute gradients

    # clw note：对于上面优化器没有使用minimize()的几点说明，
    # 使用minimize()操作，该操作不仅可以计算出梯度，而且还可以将梯度作用在变量上。
    # 如果想按照自己的方式处理梯度，可以按照以下步骤：
    # 1、使用compute_gradients()计算梯度，其实下面的get_gradients()方法就是optimizer.compute_gradients(loss)
    # 2、使用自己的方式进一步处理梯度
    # 3、使用apply_gradients()应用处理过后的梯度；

    gradients = faster_rcnn.get_gradients(optimizer, total_loss)

    # enlarge_gradients for bias
    if cfgs.MUTILPY_BIAS_GRADIENT:
        gradients = faster_rcnn.enlarge_gradients_for_bias(gradients)

    if cfgs.GRADIENT_CLIPPING_BY_NORM: # clw note：clip_by_norm是指对梯度进行裁剪，通过控制梯度的最大范式，防止梯度爆炸的问题，是一种比较常用的梯度规约的方式
        with tf.name_scope('clip_gradients_YJR'):
            gradients = slim.learning.clip_gradient_norms(gradients,
                                                          cfgs.GRADIENT_CLIPPING_BY_NORM)
    # _____________________________________________________________________________________________compute gradients


    # train_op
    train_op = optimizer.apply_gradients(grads_and_vars=gradients,
                                         global_step=global_step)
    summary_op = tf.summary.merge_all()
    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    restorer, restore_ckpt = faster_rcnn.get_restorer()
    saver = tf.train.Saver(max_to_keep=30)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        if not restorer is None:
            restorer.restore(sess, restore_ckpt)
            print('restore model')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)

        summary_path = os.path.join(cfgs.SUMMARY_PATH, cfgs.VERSION)
        # tools.mkdir(summary_path)
        if not os.path.exists(summary_path):
            os.makedirs(summary_path)
        summary_writer = tf.summary.FileWriter(summary_path, graph=sess.graph)

        for step in range(cfgs.MAX_ITERATION):
            training_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

            if step % cfgs.SHOW_TRAIN_INFO_INTE != 0 and step % cfgs.SMRY_ITER != 0:
                _, global_stepnp = sess.run([train_op, global_step])

            else:
                if step % cfgs.SHOW_TRAIN_INFO_INTE == 0 and step % cfgs.SMRY_ITER != 0:
                    start = time.time()

                    _, global_stepnp, img_name, rpnLocLoss, rpnClsLoss, rpnTotalLoss, \
                    fastrcnnLocLoss, fastrcnnClsLoss, fastrcnnTotalLoss, totalLoss = \
                        sess.run(
                            [train_op, global_step, img_name_batch, rpn_location_loss, rpn_cls_loss, rpn_total_loss,
                             fastrcnn_loc_loss, fastrcnn_cls_loss, fastrcnn_total_loss, total_loss])

                    end = time.time()
                    print(""" {}: step{}    image_name:{} |\t
                              rpn_loc_loss:{} |\t rpn_cla_loss:{} |\t rpn_total_loss:{} |
                              fast_rcnn_loc_loss:{} |\t fast_rcnn_cla_loss:{} |\t fast_rcnn_total_loss:{} |
                              total_loss:{} |\t per_cost_time:{}s""" \
                          .format(training_time, global_stepnp, str(img_name[0]), rpnLocLoss, rpnClsLoss,
                                  rpnTotalLoss, fastrcnnLocLoss, fastrcnnClsLoss, fastrcnnTotalLoss, totalLoss,
                                  (end - start)))
                else:
                    if step % cfgs.SMRY_ITER == 0:
                        _, global_stepnp, summary_str = sess.run([train_op, global_step, summary_op])
                        summary_writer.add_summary(summary_str, global_stepnp)
                        summary_writer.flush()

            if (step > 0 and step % cfgs.SAVE_WEIGHTS_INTE == 0) or (step == cfgs.MAX_ITERATION - 1):

                save_dir = os.path.join(cfgs.TRAINED_CKPT, cfgs.VERSION)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                save_ckpt = os.path.join(save_dir, 'voc_' + str(global_stepnp) + 'model.ckpt')
                saver.save(sess, save_ckpt)
                print(' weights had been saved')

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':

    train()

#
















