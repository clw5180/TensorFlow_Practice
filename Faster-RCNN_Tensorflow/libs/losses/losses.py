# -*- coding: utf-8 -*-
"""
@author: jemmy li
@contact: zengarden2009@gmail.com
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def _smooth_l1_loss_base(bbox_pred, bbox_targets, sigma=1.0):
    '''

    :param bbox_pred: [-1, 4] in RPN. [-1, cls_num+1, 4] in Fast-rcnn
    :param bbox_targets: shape is same as bbox_pred
    :param sigma:
    :return:
    '''
    sigma_2 = sigma**2

    box_diff = bbox_pred - bbox_targets

    abs_box_diff = tf.abs(box_diff)

    smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_box_diff, 1. / sigma_2)))
    # 这里得到的是根据smooth_l1 公式得到的 ，这里 x 是 bbox的差值，如果真实值与预测
    # 值之间的差值小于1/sigma_2 则 采用第一种计算方式，如果大于 1/sigma_2 则采用第二种计算方式
    # clw note：这里相当于smooth_l1的一个变种，正常按照论文里应该是：
    # if abs_box_diff < 1:
    #     loss_box = tf.pow(box_diff, 2) * 0.5
    # else:
    #     loss_box = abs_box_diff - 0.5
    loss_box = tf.pow(box_diff, 2) * (sigma_2 / 2.0) * smoothL1_sign \
               + (abs_box_diff - (0.5 / sigma_2)) * (1.0 - smoothL1_sign)
    return loss_box

# 这个代码实现的作者，并没有直接采用0.5 而是再引入了 一个sigma 因子来约束
def smooth_l1_loss_rpn(bbox_pred, bbox_targets, label, sigma=1.0):
    '''

    :param bbox_pred: [-1, 4]
    :param bbox_targets: [-1, 4]
    :param label: [-1]
    :param sigma:
    :return:
    '''
    value = _smooth_l1_loss_base(bbox_pred, bbox_targets, sigma=sigma)
    value = tf.reduce_sum(value, axis=1)  # to sum in axis 1
    # 这一块对 相应的 tx+ty+tw+th 求和，获得每一个anchor对应损失值
    rpn_select = tf.where(tf.greater(label, 0))

    # rpn_select = tf.stop_gradient(rpn_select) # to avoid
    # 在所有的value中，只选择那些为正例的样本，因为
    # 负例的样本 label 为 0 因此在公式中不参与贡献，但是最后均值的时候确实 正负一起考虑的。
    selected_value = tf.gather(value, rpn_select)
    non_ignored_mask = tf.stop_gradient(
        1.0 - tf.to_float(tf.equal(label, -1))) # positve is 1.0 others is 0.0

    bbox_loss = tf.reduce_sum(selected_value) / tf.maximum(1.0, tf.reduce_sum(non_ignored_mask))
    # 这里将选出来的正样本所有的bbox的loss求和后除以这个batch中 label为0 与
    # label为1 的样本的总数，起到了公式中除以N（pos+neg）的作用
    return bbox_loss


# 这里的输入为 bbox_pred ,是特征图上对应的roi经过Pooling，全连接层得到bbox_pred，bbox_targets是 roi对应的具有
# 最大重叠率的ground truth 框的映射因子t矩阵，labels 则是每一个target对应的类别。具体自己算Fast-RCNN的损失函数如下：
def smooth_l1_loss_rcnn(bbox_pred, bbox_targets, label, num_classes, sigma=1.0):
    '''

    :param bbox_pred: [-1, (cfgs.CLS_NUM +1) * 4]
    :param bbox_targets:[-1, (cfgs.CLS_NUM +1) * 4]
    :param label:[-1]
    :param num_classes:
    :param sigma:
    :return:
    '''

    outside_mask = tf.stop_gradient(tf.to_float(tf.greater(label, 0)))

    # 选出那些需要计算损失的roi所在的标签
    bbox_pred = tf.reshape(bbox_pred, [-1, num_classes, 4])
    # 每一个roi，对所有的类别预测位置
    bbox_targets = tf.reshape(bbox_targets, [-1, num_classes, 4])

    value = _smooth_l1_loss_base(bbox_pred,
                                 bbox_targets,
                                 sigma=sigma)
    value = tf.reduce_sum(value, 2)
    value = tf.reshape(value, [-1, num_classes])

    # 得出roi 在每个类上的预测误差
    inside_mask = tf.one_hot(tf.reshape(label, [-1, 1]),
                             depth=num_classes, axis=1)

    # 得到相关的roi的类别标签的one-hot编码
    inside_mask = tf.stop_gradient(
        tf.to_float(tf.reshape(inside_mask, [-1, num_classes])))

    normalizer = tf.to_float(tf.shape(bbox_pred)[0])
    bbox_loss = tf.reduce_sum(
        tf.reduce_sum(value * inside_mask, 1)*outside_mask) / normalizer

    # 参与计算的roi（数量为roi_per_img，定义在cfg文件中） 的平均回归误差
    return bbox_loss


def sum_ohem_loss(cls_score, label, bbox_pred, bbox_targets,
                  num_classes, num_ohem_samples=256, sigma=1.0):
    '''

    :param cls_score: [-1, cls_num+1]
    :param label: [-1]
    :param bbox_pred: [-1, 4*(cls_num+1)]
    :param bbox_targets: [-1, 4*(cls_num+1)]
    :param num_ohem_samples: 256 by default
    :param num_classes: cls_num+1
    :param sigma:
    :return:
    '''

    cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label)  # [-1, ]
    # cls_loss = tf.Print(cls_loss, [tf.shape(cls_loss)], summarize=10, message='CLS losss shape ****')

    outside_mask = tf.stop_gradient(tf.to_float(tf.greater(label, 0)))
    bbox_pred = tf.reshape(bbox_pred, [-1, num_classes, 4])
    bbox_targets = tf.reshape(bbox_targets, [-1, num_classes, 4])

    value = _smooth_l1_loss_base(bbox_pred,
                                 bbox_targets,
                                 sigma=sigma)
    value = tf.reduce_sum(value, 2)
    value = tf.reshape(value, [-1, num_classes])

    inside_mask = tf.one_hot(tf.reshape(label, [-1, 1]),
                             depth=num_classes, axis=1)

    inside_mask = tf.stop_gradient(
        tf.to_float(tf.reshape(inside_mask, [-1, num_classes])))
    loc_loss = tf.reduce_sum(value * inside_mask, 1)*outside_mask
    # loc_loss = tf.Print(loc_loss, [tf.shape(loc_loss)], summarize=10, message='loc_loss shape***')

    sum_loss = cls_loss + loc_loss

    num_ohem_samples = tf.stop_gradient(tf.minimum(num_ohem_samples, tf.shape(sum_loss)[0]))
    _, top_k_indices = tf.nn.top_k(sum_loss, k=num_ohem_samples)

    cls_loss_ohem = tf.gather(cls_loss, top_k_indices)
    cls_loss_ohem = tf.reduce_mean(cls_loss_ohem)

    loc_loss_ohem = tf.gather(loc_loss, top_k_indices)
    normalizer = tf.to_float(num_ohem_samples)
    loc_loss_ohem = tf.reduce_sum(loc_loss_ohem) / normalizer

    return cls_loss_ohem, loc_loss_ohem

