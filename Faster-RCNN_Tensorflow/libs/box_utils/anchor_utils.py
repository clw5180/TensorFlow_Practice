# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division

import tensorflow as tf

def make_anchors(base_anchor_size,  # clw note：作者给的默认值是256，如果是检测小物体，可以考虑配成64或其他值
                 anchor_scales,     #           论文中是[0.5, 1, 2]
                 anchor_ratios,     #           论文中是[0.5, 1, 2]
                 featuremap_height, #           比如224*224的图，backbone是ResNet50，到这里就相当于14*14（相当于除以16）
                 featuremap_width,  #           因为这里是从ResNet的conv_4接出来送入RPN网络，然后conv_5作为head，详见resnet.py
                                    #           具体可见笔记：ResNet结构图
                 stride,            #           cfgs.py中默认设置为16，且作者建议不要修改
                 name='make_anchors'):
    '''
    :param base_anchor_size:256
    :param anchor_scales:
    :param anchor_ratios:
    :param featuremap_height:
    :param featuremap_width:
    :param stride:
    :return:
    '''
    with tf.variable_scope(name):
        base_anchor = tf.constant([0, 0, base_anchor_size, base_anchor_size], tf.float32)  # [x_center, y_center, w, h]


        ws, hs = enum_ratios(enum_scales(base_anchor, anchor_scales),
                             anchor_ratios)  # per locations ws and hs

        x_centers = tf.range(featuremap_width, dtype=tf.float32) * stride
        y_centers = tf.range(featuremap_height, dtype=tf.float32) * stride

        x_centers, y_centers = tf.meshgrid(x_centers, y_centers)

        ws, x_centers = tf.meshgrid(ws, x_centers)
        hs, y_centers = tf.meshgrid(hs, y_centers)

        anchor_centers = tf.stack([x_centers, y_centers], 2)
        anchor_centers = tf.reshape(anchor_centers, [-1, 2])

        box_sizes = tf.stack([ws, hs], axis=2)
        box_sizes = tf.reshape(box_sizes, [-1, 2])
        # anchors = tf.concat([anchor_centers, box_sizes], axis=1)
        anchors = tf.concat([anchor_centers - 0.5*box_sizes,
                             anchor_centers + 0.5*box_sizes], axis=1)
        return anchors

#-----------------------------------------------------------------------
# clw note：具体生成框的代码分析如下：
# 经过两层枚举，就可以得到九种大小不同的hs和ws。
# 比如base_anchor_size = [256]， 对应base_anchor=[0, 0, 256, 256]，anchor_scales=[0.5, 1, 2]，
# 首先经过enum_scales()得到anchor_scales=[[128], [256], [512]]
# 之后经过enum_ratios()，比如anchor_ratios=[0.5, 1, 2]
#
# 经过乘以stride后所得到的x_center 与 y_center ,则为具体在原始图像上的锚点中心，
# 经过anchors = tf.concat([anchor_centers - 0.5*box_sizes, anchor_centers + 0.5*box_sizes], axis=1)，
# 则可以得到每一个锚点对应的九种不同anchor 的坐标值分别为，（左下角，右上角）。
def enum_scales(base_anchor, anchor_scales):

    anchor_scales = base_anchor * tf.constant(anchor_scales, dtype=tf.float32, shape=(len(anchor_scales), 1))

    return anchor_scales


def enum_ratios(anchors, anchor_ratios):
    '''
    ratio = h / w
    :param anchors:
    :param anchor_ratios:
    :return:
    '''
    ws = anchors[:, 2]  # for base anchor: w == h
    hs = anchors[:, 3]
    sqrt_ratios = tf.sqrt(tf.constant(anchor_ratios))

    ws = tf.reshape(ws / sqrt_ratios[:, tf.newaxis], [-1, 1])
    hs = tf.reshape(hs * sqrt_ratios[:, tf.newaxis], [-1, 1])

    return hs, ws
#-----------------------------------------------------------------------

