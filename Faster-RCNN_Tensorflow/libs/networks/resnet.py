# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division


import tensorflow as tf
import tensorflow.contrib.slim as slim
from libs.configs import cfgs
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_block
# import tfplot as tfp

def resnet_arg_scope(
        is_training=True,
        weight_decay=cfgs.WEIGHT_DECAY, # L2权重衰减速率
        batch_norm_decay=0.997,         # BN的衰减速率
        batch_norm_epsilon=1e-5,        # BN的epsilon默认1e-5
        batch_norm_scale=True):         # BN的scale默认值
    '''

    作者：默认我们不会使用BN来训练resnet，因为batch_size目前只支持1，太小了；
    所以在batch_norm_params直接写了'is_training': False
    自注：这个trainable搜索了一下，貌似没用到，而且官方的resnet_v1.py也没有这个参数，是作者改编的；

    '''
    batch_norm_params = {    # clw note：定义batch normalization（标准化）的参数字典
        # 是否是在训练模式，如果是在训练阶段，将会使用指数衰减函数（衰减系数为指定的decay），
        # 对moving_mean和moving_variance进行统计特性的动量更新，也就是进行使用指数衰减函数对均值和方
        # 差进行更新,而如果是在测试阶段，均值和方差就是固定不变的，是在训练阶段就求好的，在训练阶段，
        # 每个批的均值和方差的更新是加上了一个指数衰减函数，而最后求得的整个训练样本的均值和方差就是所
        # 有批的均值的均值，和所有批的方差的无偏估计
        'is_training': False,

        # 该参数能够衡量使用指数衰减函数更新均值方差时，更新的速度，取值通常在0.999-0.99-0.9之间，值
        # 越小，代表更新速度越快，而值太大的话，有可能会导致均值方差更新太慢，而最后变成一个常量1，而
        # 这个值会导致模型性能较低很多.另外，如果出现过拟合时，也可以考虑增加均值和方差的更新速度，也
        # 就是减小decay
        'decay': batch_norm_decay,

        # 就是在归一化时，除以方差时，防止方差为0而加上的一个数
        'epsilon': batch_norm_epsilon,

        'scale': batch_norm_scale,

        # 作者自己添加的一个参数，感觉还没用到
        'trainable': False,

        # force in-place updates of mean and variance estimates
        # 该参数有一个默认值，ops.GraphKeys.UPDATE_OPS，当取默认值时，slim会在当前批训练完成后再更新均
        # 值和方差，这样会存在一个问题，就是当前批数据使用的均值和方差总是慢一拍，最后导致训练出来的模
        # 型性能较差。所以，一般需要将该值设为None，这样slim进行批处理时，会对均值和方差进行即时更新，
        # 批处理使用的就是最新的均值和方差。
        #
        # 另外，不论是即使更新还是一步训练后再对所有均值方差一起更新，对测试数据是没有影响的，即测试数
        # 据使用的都是保存的模型中的均值方差数据，但是如果你在训练中需要测试，而忘了将is_training这个值
        # 改成false，那么这批测试数据将会综合当前批数据的均值方差和训练数据的均值方差。而这样做应该是不
        # 正确的。
        'updates_collections': tf.GraphKeys.UPDATE_OPS
    }

    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            trainable=is_training,
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc


# def add_heatmap(feature_maps, name):
#     '''
#
#     :param feature_maps:[B, H, W, C]
#     :return:
#     '''
#
#     def figure_attention(activation):
#         fig, ax = tfp.subplots()
#         im = ax.imshow(activation, cmap='jet')
#         fig.colorbar(im)
#         return fig
#
#     heatmap = tf.reduce_sum(feature_maps, axis=-1)
#     heatmap = tf.squeeze(heatmap, axis=0)
#     tfp.summary.plot(name, figure_attention, [heatmap])


def resnet_base(img_batch, scope_name, is_training=True):
    '''
    this code is derived from light-head rcnn.
    https://github.com/zengarden/light_head_rcnn

    It is convenient to freeze blocks. So we adapt this mode.
    '''
    if scope_name == 'resnet_v1_50':
        middle_num_units = 6
    elif scope_name == 'resnet_v1_101':
        middle_num_units = 23
    else:
        raise NotImplementedError('We only support resnet_v1_50 or resnet_v1_101. Check your network name....yjr')

    # clw note：调用slim的resnet_v1_block接口；下面的配置可以在ResNet论文中不同层数时的网络配置查到
    #           对于ResNet_v1_50，为 1（conv1）+ 3 * 3（conv2）+ 4 * 3（conv3）+ 6 * 3（conv4）+
    #           # 3 * 3（conv5） = 1+9+12+18+9+1fc=50
    blocks = [resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
              resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
              # use stride 1 for the last conv4 layer.

              resnet_v1_block('block3', base_depth=256, num_units=middle_num_units, stride=1)]
              # when use fpn . stride list is [1, 2, 2]

    with slim.arg_scope(resnet_arg_scope(is_training=False)):
        with tf.variable_scope(scope_name, scope_name):
            # Do the first few layers manually, because 'SAME' padding can behave inconsistently
            # for images of different sizes: sometimes 0, sometimes 1
            net = resnet_utils.conv2d_same(
                img_batch, 64, 7, stride=2, scope='conv1')
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.max_pool2d(
                net, [3, 3], stride=2, padding='VALID', scope='pool1')

    # clw note：在resnet.py文件中，定义了resenet_base网络以及resnet_head网络，一个作为基础的特征提取网络，
    # 另一个则作为RoI Pooling后的检测，分类顶层网络。
    # 在建立base网络时，根据下面的not_freezed
    # 确定是否对特征提取网络进行再训练
    not_freezed = [False] * cfgs.FIXED_BLOCKS + (4-cfgs.FIXED_BLOCKS)*[True]
    # Fixed_Blocks can be 1~3

    with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[0]))):
        C2, _ = resnet_v1.resnet_v1(net,
                                    blocks[0:1],
                                    global_pool=False,
                                    include_root_block=False,
                                    scope=scope_name)

    # C2 = tf.Print(C2, [tf.shape(C2)], summarize=10, message='C2_shape')
    # add_heatmap(C2, 'Layer/C2')

    with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[1]))):
        C3, _ = resnet_v1.resnet_v1(C2,
                                    blocks[1:2],
                                    global_pool=False,
                                    include_root_block=False,
                                    scope=scope_name)
    # add_heatmap(C3, name='Layer/C3')
    # C3 = tf.Print(C3, [tf.shape(C3)], summarize=10, message='C3_shape')

    with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[2]))):
        C4, _ = resnet_v1.resnet_v1(C3,
                                    blocks[2:3],
                                    global_pool=False,
                                    include_root_block=False,
                                    scope=scope_name)
    # add_heatmap(C4, name='Layer/C4')
    # C4 = tf.Print(C4, [tf.shape(C4)], summarize=10, message='C4_shape')
    return C4


def restnet_head(input, is_training, scope_name): # clw note：resnet_head是在更深层的位置
    block4 = [resnet_v1_block('block4', base_depth=512, num_units=3, stride=1)]

    with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        C5, _ = resnet_v1.resnet_v1(input,
                                    block4,
                                    global_pool=False,
                                    include_root_block=False,
                                    scope=scope_name)
        # C5 = tf.Print(C5, [tf.shape(C5)], summarize=10, message='C5_shape')
        C5_flatten = tf.reduce_mean(C5, axis=[1, 2], keep_dims=False, name='global_average_pooling')
        # C5_flatten = tf.Print(C5_flatten, [tf.shape(C5_flatten)], summarize=10, message='C5_flatten_shape')

    # global average pooling C5 to obtain fc layers
    return C5_flatten































