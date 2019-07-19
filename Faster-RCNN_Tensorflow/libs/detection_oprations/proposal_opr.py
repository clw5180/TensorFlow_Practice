# encoding: utf-8
"""
@author: zeming li
@contact: zengarden2009@gmail.com
"""

from libs.configs import cfgs
from libs.box_utils import encode_and_decode
from libs.box_utils import boxes_utils
import tensorflow as tf
import numpy as np


def postprocess_rpn_proposals(rpn_bbox_pred, rpn_cls_prob, img_shape, anchors, is_training):
    '''

    :param rpn_bbox_pred: [-1, 4]
    :param rpn_cls_prob: [-1, 2]
    :param img_shape:
    :param anchors:[-1, 4]
    :param is_training:
    :return:
    '''

    if is_training:
        pre_nms_topN = cfgs.RPN_TOP_K_NMS_TRAIN          # 默认12000
        post_nms_topN = cfgs.RPN_MAXIMUM_PROPOSAL_TARIN  # 默认2000
        nms_thresh = cfgs.RPN_NMS_IOU_THRESHOLD          # 默认0.7
    else:
        pre_nms_topN = cfgs.RPN_TOP_K_NMS_TEST           # 默认6000
        post_nms_topN = cfgs.RPN_MAXIMUM_PROPOSAL_TEST   # 默认300
        nms_thresh = cfgs.RPN_NMS_IOU_THRESHOLD          # 默认0.7

    cls_prob = rpn_cls_prob[:, 1]

    # 1. decode boxes
    # clw note：这个函数接受RPN网络的预测框位置，以及预测的类别（两类），图像的尺寸大小，以及生成的锚点作为输入。
    #           经过解码后，得到的是真实的预测框的位置，因为有可能预测的框比设定的选取前N个框的个数还小，
    #           因此在预测框的数目以及设定的数目之间取最小值，之后再采用 tf.image.non_max_suppression抑制，
    #           选取最终的非极大值抑制后的Top K个框，原论文中未采用NMS之前为12000个（就是上面的cfgs.RPN_TOP_K_NMS_TRAIN），
    #           NMS后为2000个（就是上面的cfgs.RPN_MAXIMUM_PROPOSAL_TARIN）。
    #           这里还没有具体的分类那个框是那个目标，只是选出了前K个可能存在目标的框。
    decode_boxes = encode_and_decode.decode_boxes(encoded_boxes=rpn_bbox_pred,
                                                  reference_boxes=anchors,
                                                  scale_factors=cfgs.ANCHOR_SCALE_FACTORS)

    # decode_boxes = encode_and_decode.decode_boxes(boxes=anchors,
    #                                               deltas=rpn_bbox_pred,
    #                                               scale_factor=None)

    # 2. clip to img boundaries
    decode_boxes = boxes_utils.clip_boxes_to_img_boundaries(decode_boxes=decode_boxes,
                                                            img_shape=img_shape)

    # 3. get top N to NMS
    if pre_nms_topN > 0:  # clw note：初步得到一系列框（~60*40*9=20k）之后，如果是训练集，会去掉与边界相交的anchors，因此
                          #           数量会大大减小，即NMS之前的TopK个框（这里默认值是12k，文中给的6k），之后再进行NMS。
        pre_nms_topN = tf.minimum(pre_nms_topN, tf.shape(decode_boxes)[0], name='avoid_unenough_boxes')
        cls_prob, top_k_indices = tf.nn.top_k(cls_prob, k=pre_nms_topN)
        decode_boxes = tf.gather(decode_boxes, top_k_indices)

    # 4. NMS
    keep = tf.image.non_max_suppression(
        boxes=decode_boxes,
        scores=cls_prob,
        max_output_size=post_nms_topN,
        iou_threshold=nms_thresh)

    final_boxes = tf.gather(decode_boxes, keep)
    final_probs = tf.gather(cls_prob, keep)

    return final_boxes, final_probs

