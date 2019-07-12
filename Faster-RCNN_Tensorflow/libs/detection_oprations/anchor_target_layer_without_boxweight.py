# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from libs.configs import cfgs
import numpy as np
import numpy.random as npr
from libs.box_utils.cython_utils.cython_bbox import bbox_overlaps
from libs.box_utils import encode_and_decode


def anchor_target_layer(
        gt_boxes, img_shape, all_anchors, is_restrict_bg=False):
    """Same as the anchor target layer in original Fast/er RCNN """

    total_anchors = all_anchors.shape[0]
    img_h, img_w = img_shape[1], img_shape[2]
    gt_boxes = gt_boxes[:, :-1]  # remove class label

    # allow boxes to sit over the edge by a small amount
    _allowed_border = 0

    # only keep anchors inside the image
    inds_inside = np.where(
        (all_anchors[:, 0] >= -_allowed_border) &
        (all_anchors[:, 1] >= -_allowed_border) &
        (all_anchors[:, 2] < img_w + _allowed_border) &  # width
        (all_anchors[:, 3] < img_h + _allowed_border)  # height
    )[0]

    anchors = all_anchors[inds_inside, :]

    # label: 1 is positive, 0 is negative, -1 is dont care
    # 首先将所有的label都定义为 - 1
    # 其label长度为在图像内部的Anchor的数目值
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)

    # overlaps between the anchors and the gt boxes
    overlaps = bbox_overlaps(
        np.ascontiguousarray(anchors, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float))

    # 计算每一行的重叠率最大的值所在的索引，行数则为在图像大小范围内的所有Anchors数目(每一个Anchor与哪一个ground truth框重叠最大
    argmax_overlaps = overlaps.argmax(axis=1)

    #取出与相关的Anchors重叠最大的ground truth的那个值
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]

    #计算出每一列的最大值的索引，一共有ground truth目标数目个列(每一个ground truth与哪一个Anchor重叠最大）
    gt_argmax_overlaps = overlaps.argmax(axis=0)

    #取出与ground truth最大重叠的Anchor的重叠率的数值
    gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

    # 如果每一个最大重叠框与其最大的ground truth框的重叠率小于RPN_IOU_NEG 的重叠率，则这个框的label为背景
    if not cfgs.TRAIN_RPN_CLOOBER_POSITIVES:
        labels[max_overlaps < cfgs.RPN_IOU_NEGATIVE_THRESHOLD] = 0

    # 如果每一个ground truth框对应的anchor的重叠率大于RPN_IOU_POS 的重叠率，则这个框的label为目标
    labels[gt_argmax_overlaps] = 1
    # 如果每一个anchor对应的最大重叠框的重叠率大于RPN_POS的重叠率阈值，则也认为其为目标
    labels[max_overlaps >= cfgs.RPN_IOU_POSITIVE_THRESHOLD] = 1

    if cfgs.TRAIN_RPN_CLOOBER_POSITIVES:
        labels[max_overlaps < cfgs.RPN_IOU_NEGATIVE_THRESHOLD] = 0

    # 预先设定的前景的目标数目
    num_fg = int(cfgs.RPN_MINIBATCH_SIZE * cfgs.RPN_POSITIVE_RATE)
    fg_inds = np.where(labels == 1)[0] # 所有label为1的包含目标的点
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(
            fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        labels[disable_inds] = -1

    # 如果label等于目标的数目大于所预先设定的目标数目的值，就随机的将部分label设定为-1，不参与计算
    num_bg = cfgs.RPN_MINIBATCH_SIZE - np.sum(labels == 1)
    if is_restrict_bg:
        num_bg = max(num_bg, num_fg * 1.5)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(
            bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1
    # 如果背景的label数目大于所设定的背景数目，则将部分的背景标签设置为 - 1，不参与计算。
    # 如果小于，则不做任何改变，保留所有背景的相关标签为0

    bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])

    # 这一块输入的参数为所有的Anchors以及与每一个anchor对应的重叠率最大的那个ground truth目标框所对应的坐标
    # bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])
    # 其返回值为每一个在图像内的anchor与其对应的具有最大重叠率的ground truth框之间的映射关系，也就是对其进行编码的过程
    #
    #
    # 因为一直在计算中都是针对于所有在图像内的框进行运算，并没有考虑到在图像外的框，但是在最终的计算中，针对的是所有的anchor，
    # 因此需要将处理过的与原始的进行融合
    # map up to original set of anchors
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)

    # labels = labels.reshape((1, height, width, A))
    rpn_labels = labels.reshape((-1, 1))

    # bbox_targets
    bbox_targets = bbox_targets.reshape((-1, 4))
    rpn_bbox_targets = bbox_targets

    # 最后返回的为编码后的label，以及映射因子矩阵
    return rpn_labels, rpn_bbox_targets


def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""
    # targets = bbox_transform(ex_rois, gt_rois[:, :4]).astype(
    #     np.float32, copy=False)
    targets = encode_and_decode.encode_boxes(unencode_boxes=gt_rois,
                                             reference_boxes=ex_rois,
                                             scale_factors=cfgs.ANCHOR_SCALE_FACTORS)
    # targets = encode_and_decode.encode_boxes(ex_rois=ex_rois,
    #                                          gt_rois=gt_rois,
    #                                          scale_factor=None)
    return targets
