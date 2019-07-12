# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from libs.configs import cfgs
import numpy as np
import numpy.random as npr

from libs.box_utils import encode_and_decode
from libs.box_utils.cython_utils.cython_bbox import bbox_overlaps


def proposal_target_layer(rpn_rois, gt_boxes):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """
    # Proposal ROIs (x1, y1, x2, y2) coming from RPN
    # gt_boxes (x1, y1, x2, y2, label)
    if cfgs.ADD_GTBOXES_TO_TRAIN:
        all_rois = np.vstack((rpn_rois, gt_boxes[:, :-1]))
    else:
        all_rois = rpn_rois
    # np.inf
    rois_per_image = np.inf if cfgs.FAST_RCNN_MINIBATCH_SIZE == -1 else cfgs.FAST_RCNN_MINIBATCH_SIZE

    fg_rois_per_image = np.round(cfgs.FAST_RCNN_POSITIVE_RATE * rois_per_image)

    # Sample rois with classification labels and bounding box regression
    # 其函数的输入为所有的由RPN所产生的RoIs，以及所有的gt_boxs,每一幅图需要产生的目标roi数目，没幅图的roi，真实的类别数目)
    labels, rois, bbox_targets = _sample_rois(all_rois, gt_boxes, fg_rois_per_image,
                                              rois_per_image, cfgs.CLASS_NUM+1)
    # print(labels.shape, rois.shape, bbox_targets.shape)
    rois = rois.reshape(-1, 4)
    labels = labels.reshape(-1)
    bbox_targets = bbox_targets.reshape(-1, (cfgs.CLASS_NUM+1) * 4)

    return rois, labels, bbox_targets


def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
    """

    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = int(4 * cls)
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]

    return bbox_targets


def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image.
    that is : [label, tx, ty, tw, th]
    """

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = encode_and_decode.encode_boxes(unencode_boxes=gt_rois,
                                             reference_boxes=ex_rois,
                                             scale_factors=cfgs.ROI_SCALE_FACTORS)
    # targets = encode_and_decode.encode_boxes(ex_rois=ex_rois,
    #                                          gt_rois=gt_rois,
    #                                          scale_factor=cfgs.ROI_SCALE_FACTORS)

    return np.hstack(
        (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)


def _sample_rois(all_rois, gt_boxes, fg_rois_per_image,
                 rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.

    all_rois shape is [-1, 4]
    gt_boxes shape is [-1, 5]. that is [x1, y1, x2, y2, label]
    """
    # overlaps: (rois x gt_boxes)

    # clw note：计算所有的RPN产生的ROI与所有的ground truth的目标框的重叠率
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois, dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :-1], dtype=np.float))

    # 得到与每一个roi最大重叠的gt_box 的框的索引 以及 重叠率
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    # 获得相对应的类别标签
    labels = gt_boxes[gt_assignment, -1]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfgs.FAST_RCNN_IOU_POSITIVE_THRESHOLD)[0]

    # Guard against the case when an image has fewer than fg_rois_per_image
    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < cfgs.FAST_RCNN_IOU_POSITIVE_THRESHOLD) &
                       (max_overlaps >= cfgs.FAST_RCNN_IOU_NEGATIVE_THRESHOLD))[0]
    # print("first fileter, fg_size: {} || bg_size: {}".format(fg_inds.shape, bg_inds.shape))
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)
    # 以最小的 fg_size 作为fg_rois_per_this_image

    # Sample foreground regions without replacement
    if fg_inds.size > 0:  # 如果有目标
        fg_inds = npr.choice(fg_inds, size=int(fg_rois_per_this_image), replace=False)
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=int(bg_rois_per_this_image), replace=False)

    # print("second fileter, fg_size: {} || bg_size: {}".format(fg_inds.shape, bg_inds.shape))
    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    #  选择出来的fg以及bg是在相关的阈值基础上得到的，bg的选取有一个最低的阈值

    # Select sampled values from various arrays:
    labels = labels[keep_inds]

    # Clamp labels for the background RoIs to 0
    labels[int(fg_rois_per_this_image):] = 0
    rois = all_rois[keep_inds]

    # 计算bbox目标数据，输入都是对应的keep_inds所对应的roi，gt_box，labels
    bbox_target_data = _compute_targets(
        rois, gt_boxes[gt_assignment[keep_inds], :-1], labels)
    # 其返回值为 roi与gt_box 之间映射的因子矩阵以及对应的类别信息,
    # 下面的函数将为每一个非background的类写入相关的四个坐标因此t，
    # 这里，由于num_classes是从tf-record 中直接得到的，因此类数量是包含background的，因此比真实的要多出一类
    bbox_targets = _get_bbox_regression_labels(bbox_target_data, num_classes)

    # 返回值后期计算的labels（这里为具体的类），rois为要保留的roi，bbox_targets 为每一个具体的类
    # （一共的NUM_CLASS个类，每一个类对应四个坐标点）对应的坐标映射矩阵
    return labels, rois, bbox_targets
