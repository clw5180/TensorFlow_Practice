# -*-coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from libs.networks import resnet
from libs.networks import mobilenet_v2
from libs.box_utils import encode_and_decode
from libs.box_utils import boxes_utils
from libs.box_utils import anchor_utils
from libs.configs import cfgs
from libs.losses import losses
from libs.losses import tfapi_loss
from libs.box_utils import show_box_in_tensor
from libs.detection_oprations.proposal_opr import postprocess_rpn_proposals
from libs.detection_oprations.anchor_target_layer_without_boxweight import anchor_target_layer
from libs.detection_oprations.proposal_target_layer import proposal_target_layer


class DetectionNetwork(object):

    def __init__(self, base_network_name, is_training):

        self.base_network_name = base_network_name
        self.is_training = is_training
        self.num_anchors_per_location = len(cfgs.ANCHOR_SCALES) * len(cfgs.ANCHOR_RATIOS)

    def build_base_network(self, input_img_batch):

        # clw note：作者当前只针对resnet_v1和MobileNetV2做了实现，因此只支持这两种网络。
        # TODO：看后续能不能添加resnet_v2或者其他更高端的CNN结构
        if self.base_network_name.startswith('resnet_v1'):
            return resnet.resnet_base(input_img_batch, scope_name=self.base_network_name, is_training=self.is_training)

        elif self.base_network_name.startswith('MobilenetV2'):
            return mobilenet_v2.mobilenetv2_base(input_img_batch, is_training=self.is_training)

        else:
            raise ValueError('Sry, we only support resnet or mobilenet_v2')

    def postprocess_fastrcnn(self, rois, bbox_ppred, scores, img_shape):
        '''

        :param rois:[-1, 4]
        :param bbox_ppred: [-1, (cfgs.Class_num+1) * 4]
        :param scores: [-1, cfgs.Class_num + 1]
        :return:
        '''

        with tf.name_scope('postprocess_fastrcnn'):
            rois = tf.stop_gradient(rois)
            scores = tf.stop_gradient(scores)
            bbox_ppred = tf.reshape(bbox_ppred, [-1, cfgs.CLASS_NUM + 1, 4])
            bbox_ppred = tf.stop_gradient(bbox_ppred)

            bbox_pred_list = tf.unstack(bbox_ppred, axis=1)
            score_list = tf.unstack(scores, axis=1)

            allclasses_boxes = []
            allclasses_scores = []
            categories = []
            for i in range(1, cfgs.CLASS_NUM+1):

                # 1. decode boxes in each class
                tmp_encoded_box = bbox_pred_list[i]
                tmp_score = score_list[i]
                tmp_decoded_boxes = encode_and_decode.decode_boxes(encoded_boxes=tmp_encoded_box,
                                                                   reference_boxes=rois,
                                                                   scale_factors=cfgs.ROI_SCALE_FACTORS)
                # tmp_decoded_boxes = encode_and_decode.decode_boxes(boxes=rois,
                #                                                    deltas=tmp_encoded_box,
                #                                                    scale_factor=cfgs.ROI_SCALE_FACTORS)

                # 2. clip to img boundaries
                tmp_decoded_boxes = boxes_utils.clip_boxes_to_img_boundaries(decode_boxes=tmp_decoded_boxes,
                                                                             img_shape=img_shape)

                # 3. NMS
                keep = tf.image.non_max_suppression(boxes=tmp_decoded_boxes,
                                                    scores=tmp_score,
                                                    max_output_size=cfgs.FAST_RCNN_NMS_MAX_BOXES_PER_CLASS,
                                                    iou_threshold=cfgs.FAST_RCNN_NMS_IOU_THRESHOLD)


                perclass_boxes = tf.gather(tmp_decoded_boxes, keep)
                perclass_scores = tf.gather(tmp_score, keep)

                allclasses_boxes.append(perclass_boxes)
                allclasses_scores.append(perclass_scores)
                categories.append(tf.ones_like(perclass_scores) * i)

            final_boxes = tf.concat(allclasses_boxes, axis=0)
            final_scores = tf.concat(allclasses_scores, axis=0)
            final_category = tf.concat(categories, axis=0)

            if self.is_training:
                '''
                in training. We should show the detecitons in the tensorboard. So we add this.
                '''
                kept_indices = tf.reshape(tf.where(tf.greater_equal(final_scores, cfgs.SHOW_SCORE_THRSHOLD)), [-1])

                final_boxes = tf.gather(final_boxes, kept_indices)
                final_scores = tf.gather(final_scores, kept_indices)
                final_category = tf.gather(final_category, kept_indices)

        return final_boxes, final_scores, final_category

    def roi_pooling(self, feature_maps, rois, img_shape):
        '''
        Here use roi warping as roi_pooling

        :param featuremaps_dict: feature map to crop
        :param rois: shape is [-1, 4]. [x1, y1, x2, y2]
        :return:
        '''

        with tf.variable_scope('ROI_Warping'):
            img_h, img_w = tf.cast(img_shape[1], tf.float32), tf.cast(img_shape[2], tf.float32)
            N = tf.shape(rois)[0]
            x1, y1, x2, y2 = tf.unstack(rois, axis=1)

            normalized_x1 = x1 / img_w
            normalized_x2 = x2 / img_w
            normalized_y1 = y1 / img_h
            normalized_y2 = y2 / img_h

            # 获得一个正则化的roi范围
            normalized_rois = tf.transpose(
                tf.stack([normalized_y1, normalized_x1, normalized_y2, normalized_x2]), name='get_normalized_rois')

            normalized_rois = tf.stop_gradient(normalized_rois)

            # 这一块还不太理解，是对正则化的roi进行梯度裁剪吗？
            cropped_roi_features = tf.image.crop_and_resize(feature_maps, normalized_rois,
                                                            box_ind=tf.zeros(shape=[N, ],
                                                                             dtype=tf.int32),
                                                            crop_size=[cfgs.ROI_SIZE, cfgs.ROI_SIZE],
                                                            name='CROP_AND_RESIZE'
                                                            )

            # 在特征图上获得与原始的图像想对应的特征图切片，并将特征图切片resize为规整的大小，以便于后期的特征图池化
            roi_features = slim.max_pool2d(cropped_roi_features,
                                           [cfgs.ROI_POOL_KERNEL_SIZE, cfgs.ROI_POOL_KERNEL_SIZE],
                                           stride=cfgs.ROI_POOL_KERNEL_SIZE)
        # 返回池化后的特征图
        return roi_features


    def build_fastrcnn(self, feature_to_cropped, rois, img_shape):
        with tf.variable_scope('Fast-RCNN'):
            # 4. ROI Pooling
            with tf.variable_scope('rois_pooling'):
                pooled_features = self.roi_pooling(feature_maps=feature_to_cropped, rois=rois, img_shape=img_shape)

            #  clw note：之后则是对特征图通过ResNet前向传播，并通过一个两个全连接网络，一个全连接网络负责做分类，
            #            另一个全连接输出的回归的坐标信息。
            # 5. inferecne rois in Fast-RCNN to obtain fc_flatten features
            if self.base_network_name.startswith('resnet'):
                fc_flatten = resnet.restnet_head(input=pooled_features,
                                                 is_training=self.is_training,
                                                 scope_name=self.base_network_name)
            elif self.base_network_name.startswith('Mobile'):
                fc_flatten = mobilenet_v2.mobilenetv2_head(inputs=pooled_features,
                                                           is_training=self.is_training)
            else:
                raise NotImplementedError('only support resnet and mobilenet')

            # 6. cls and reg in Fast-RCNN
            # tf.variance_scaling_initializer()
            # tf.VarianceScaling()
            # 顶层的网络线通过的了一个ResNet的残差网络，最后将残差网络的输出分别接上两FC网络作为分类和回归的输出，
            # 此时回归的输出依然为映射因子，后期需要对其进行decode才能转变为正常的在图像中的区域。以下为两层FC网络。
            with slim.arg_scope([slim.fully_connected], weights_regularizer=slim.l2_regularizer(cfgs.WEIGHT_DECAY)):

                # 分类值
                cls_score = slim.fully_connected(fc_flatten,
                                                 num_outputs=cfgs.CLASS_NUM+1,
                                                 weights_initializer=slim.variance_scaling_initializer(factor=1.0,
                                                                                                       mode='FAN_AVG',
                                                                                                       uniform=True),
                                                 activation_fn=None, trainable=self.is_training,
                                                 scope='cls_fc')
                # 预测目标框值，输出为类数目*4
                bbox_pred = slim.fully_connected(fc_flatten,
                                                 num_outputs=(cfgs.CLASS_NUM+1)*4,
                                                 weights_initializer=slim.variance_scaling_initializer(factor=1.0,
                                                                                                       mode='FAN_AVG',
                                                                                                       uniform=True),
                                                 activation_fn=None, trainable=self.is_training,
                                                 scope='reg_fc')
                # for convient. It also produce (cls_num +1) bboxes

                cls_score = tf.reshape(cls_score, [-1, cfgs.CLASS_NUM+1])
                bbox_pred = tf.reshape(bbox_pred, [-1, 4*(cfgs.CLASS_NUM+1)])

        return bbox_pred, cls_score

    def add_anchor_img_smry(self, img, anchors, labels):

        positive_anchor_indices = tf.reshape(tf.where(tf.greater_equal(labels, 1)), [-1])
        negative_anchor_indices = tf.reshape(tf.where(tf.equal(labels, 0)), [-1])

        positive_anchor = tf.gather(anchors, positive_anchor_indices)
        negative_anchor = tf.gather(anchors, negative_anchor_indices)

        pos_in_img = show_box_in_tensor.only_draw_boxes(img_batch=img,
                                                        boxes=positive_anchor)
        neg_in_img = show_box_in_tensor.only_draw_boxes(img_batch=img,
                                                        boxes=negative_anchor)

        tf.summary.image('positive_anchor', pos_in_img)
        tf.summary.image('negative_anchors', neg_in_img)

    def add_roi_batch_img_smry(self, img, rois, labels):
        positive_roi_indices = tf.reshape(tf.where(tf.greater_equal(labels, 1)), [-1])

        negative_roi_indices = tf.reshape(tf.where(tf.equal(labels, 0)), [-1])

        pos_roi = tf.gather(rois, positive_roi_indices)
        neg_roi = tf.gather(rois, negative_roi_indices)


        pos_in_img = show_box_in_tensor.only_draw_boxes(img_batch=img,
                                                               boxes=pos_roi)
        neg_in_img = show_box_in_tensor.only_draw_boxes(img_batch=img,
                                                               boxes=neg_roi)
        tf.summary.image('pos_rois', pos_in_img)
        tf.summary.image('neg_rois', neg_in_img)

    def build_loss(self, rpn_box_pred, rpn_bbox_targets, rpn_cls_score, rpn_labels,
                   bbox_pred, bbox_targets, cls_score, labels):
        '''

        :param rpn_box_pred: [-1, 4]
        :param rpn_bbox_targets: [-1, 4]
        :param rpn_cls_score: [-1]
        :param rpn_labels: [-1]
        :param bbox_pred: [-1, 4*(cls_num+1)]
        :param bbox_targets: [-1, 4*(cls_num+1)]
        :param cls_score: [-1, cls_num+1]
        :param labels: [-1]
        :return:
        '''
        with tf.variable_scope('build_loss') as sc:
            with tf.variable_scope('rpn_loss'):

                # 利用smooth_l1_loss_rpn 函数计算边框回归误差
                rpn_bbox_loss = losses.smooth_l1_loss_rpn(bbox_pred=rpn_box_pred,
                                                          bbox_targets=rpn_bbox_targets,
                                                          label=rpn_labels,
                                                          sigma=cfgs.RPN_SIGMA)
                # rpn_bbox_loss = tfapi_loss.smooth_l1_loss_rpn(bbox_pred=rpn_box_pred,
                #                                               bbox_targets=rpn_bbox_targets,
                #                                               label=rpn_labels,
                #                                               sigma=cfgs.RPN_SIGMA)
                # rpn_cls_loss:
                # rpn_cls_score = tf.reshape(rpn_cls_score, [-1, 2])
                # rpn_labels = tf.reshape(rpn_labels, [-1])
                # ensure rpn_labels shape is [-1]
                # 只选出那些 label 为 0 与 1 的框label 所对应的 rpn_cls_score
                rpn_select = tf.reshape(tf.where(tf.not_equal(rpn_labels, -1)), [-1])
                rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
                # 选出那些label所对应的 ground_truth 对应的标签 函数接收的时候为所有的anchor对应的，针对label的赋值，来小批量的处理
                rpn_labels = tf.reshape(tf.gather(rpn_labels, rpn_select), [-1])

                # 采用交叉熵衡量分类误差，这里只有 有目标与没目标之间两类
                rpn_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score,
                                                                                             labels=rpn_labels))
                # 对分类以及回归误差分别进行加权后return
                rpn_cls_loss = rpn_cls_loss * cfgs.RPN_CLASSIFICATION_LOSS_WEIGHT
                rpn_bbox_loss = rpn_bbox_loss * cfgs.RPN_LOCATION_LOSS_WEIGHT

            # 首先从所有的rpn_label 中选出仅仅作为minibatch参与计算loss的label，获得相关的rpn网络对于minibatch中anchor
            # 的二分类值，最后计算标签与预测值之间的交叉熵，得到分类的误差，得到这两类误差后，就可以对其进行加权相加得到
            # RPN网络的总误差值。在计算Fast-RCNN的边框回归误差以及分类误差的时候，由于RPN与Fast-RCNN采用的loss相同，
            # 只是在具体的类别上不一致，因此采用了与RPN网络同样的误差函数：
            with tf.variable_scope('FastRCNN_loss'):
                if not cfgs.FAST_RCNN_MINIBATCH_SIZE == -1:
                    bbox_loss = losses.smooth_l1_loss_rcnn(bbox_pred=bbox_pred,
                                                           bbox_targets=bbox_targets,
                                                           label=labels,
                                                           num_classes=cfgs.CLASS_NUM + 1,
                                                           sigma=cfgs.FASTRCNN_SIGMA)
                    # bbox_loss = tfapi_loss.smooth_l1_loss_rcnn(bbox_pred=bbox_pred,
                    #                                            bbox_targets=bbox_targets,
                    #                                            label=labels,
                    #                                            num_classes=cfgs.CLASS_NUM + 1,
                    #                                            sigma=cfgs.FASTRCNN_SIGMA)

                    # cls_score = tf.reshape(cls_score, [-1, cfgs.CLASS_NUM + 1])
                    # labels = tf.reshape(labels, [-1])
                    cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=cls_score,
                        labels=labels))  # beacause already sample before
                else:
                    ''' 
                    applying OHEM here
                    '''
                    print(20 * "@@")
                    print("@@" + 10 * " " + "TRAIN WITH OHEM ...")
                    print(20 * "@@")
                    cls_loss, bbox_loss = losses.sum_ohem_loss(cls_score=cls_score,
                                                               label=labels,
                                                               bbox_targets=bbox_targets,
                                                               bbox_pred=bbox_pred,
                                                               num_ohem_samples=256,
                                                               num_classes=cfgs.CLASS_NUM + 1)
                # 在计算Fast-RCNN的分类误差时，采用所有类的交叉熵，得到回归误差与分类误差加权。
                cls_loss = cls_loss * cfgs.FAST_RCNN_CLASSIFICATION_LOSS_WEIGHT
                bbox_loss = bbox_loss * cfgs.FAST_RCNN_LOCATION_LOSS_WEIGHT
            loss_dict = {
                'rpn_cls_loss': rpn_cls_loss,
                'rpn_loc_loss': rpn_bbox_loss,
                'fastrcnn_cls_loss': cls_loss,
                'fastrcnn_loc_loss': bbox_loss
            }
        return loss_dict

    # clw note：最重要的函数，构建Faster R-CNN网络结构；
    #           其输入为批次图像，以及相关批次的ground truth
    def build_whole_detection_network(self, input_img_batch, gtboxes_batch):

        if self.is_training:
            # ensure shape is [M, 5]
            gtboxes_batch = tf.reshape(gtboxes_batch, [-1, 5])
            gtboxes_batch = tf.cast(gtboxes_batch, tf.float32)

        img_shape = tf.shape(input_img_batch) # 网络首先获得批次数据的基本信息，图像的shape 等。

        # 1. build base network
        feature_to_cropped = self.build_base_network(input_img_batch) # 首先建立基础特征提取网络

        # 2. build rpn
        with tf.variable_scope('build_rpn',  # clw note：变量空间的名称，tf.variable_scope()主要用于管理图中变量的名字
                                             #           而tf.name_scope()主要用于管理图中各种op；
                               regularizer=slim.l2_regularizer(cfgs.WEIGHT_DECAY)): # 默认的正则化函数

            # 建立RPN网络，可以看出，这是一个三层的卷积网络，第一层网络采用一个3X3的卷积核在特征图上滑动，生成一个高层的特征图，
            # 第二层的一路通过1x1的卷积核，stride=1进行滑动，每一处的输出维度为锚点数*2（每一处是否为目标 相当于二分类）
            # 输出维度与rpn_conv3x3相同，
            # 另一路在卷积特征图上用1x1的卷积核stride为1进行卷积，卷积核的深度为锚点数 * 4。根据原始faster - rcnn论文，
            # 这里第二层一路卷积输出为当前位置是否含有目标，另一路卷积输出为框回归坐标，第二层卷积核通过softmax函数归一化处理。
            rpn_conv3x3 = slim.conv2d(feature_to_cropped, 512, [3, 3],
                                      trainable=self.is_training,
                                      weights_initializer=cfgs.INITIALIZER,
                                      activation_fn=tf.nn.relu,
                                      scope='rpn_conv/3x3')
            rpn_cls_score = slim.conv2d(rpn_conv3x3,
                                        self.num_anchors_per_location*2, [1, 1],
                                        stride=1,
                                        trainable=self.is_training,
                                        weights_initializer=cfgs.INITIALIZER,
                                        activation_fn=None,
                                        scope='rpn_cls_score')
            rpn_box_pred = slim.conv2d(rpn_conv3x3, self.num_anchors_per_location*4, [1, 1], stride=1,
                                       trainable=self.is_training, weights_initializer=cfgs.BBOX_INITIALIZER,
                                       activation_fn=None,
                                       scope='rpn_bbox_pred')
            rpn_box_pred = tf.reshape(rpn_box_pred, [-1, 4])
            rpn_cls_score = tf.reshape(rpn_cls_score, [-1, 2])
            rpn_cls_prob = slim.softmax(rpn_cls_score, scope='rpn_cls_prob')


        # --------------- 3. generate_anchors ---------------

        # clw note：在resnet_base网络所获得的特征图上根据scales以及ratios产生Anchors
        featuremap_height, featuremap_width = tf.shape(feature_to_cropped)[1], tf.shape(feature_to_cropped)[2]
        featuremap_height = tf.cast(featuremap_height, tf.float32)
        featuremap_width = tf.cast(featuremap_width, tf.float32)

        #  clw note：调用make_anchors()函数，在resnet_base 网络所获得的特征图上根据scales以及ratios产生Anchors
        anchors = anchor_utils.make_anchors(base_anchor_size=cfgs.BASE_ANCHOR_SIZE_LIST[0],
                                            anchor_scales=cfgs.ANCHOR_SCALES, anchor_ratios=cfgs.ANCHOR_RATIOS,
                                            featuremap_height=featuremap_height,
                                            featuremap_width=featuremap_width,
                                            stride=cfgs.ANCHOR_STRIDE,
                                            name="make_anchors_forRPN")

        # with tf.variable_scope('make_anchors'):
        #     anchors = anchor_utils.make_anchors(height=featuremap_height,
        #                                         width=featuremap_width,
        #                                         feat_stride=cfgs.ANCHOR_STRIDE[0],
        #                                         anchor_scales=cfgs.ANCHOR_SCALES,
        #                                         anchor_ratios=cfgs.ANCHOR_RATIOS, base_size=16
        #                                         )


        # --------------- 4. postprocess rpn proposals. such as: decode, clip, NMS ---------------

        # clw note： 对于RPN 进行预处理，编码，切片，非极大值抑制（NMS）
        with tf.variable_scope('postprocess_RPN'):
            # rpn_cls_prob = tf.reshape(rpn_cls_score, [-1, 2])
            # rpn_cls_prob = slim.softmax(rpn_cls_prob, scope='rpn_cls_prob')
            # rpn_box_pred = tf.reshape(rpn_box_pred, [-1, 4])

            # clw note：此时生成的anchor是没有经过任何处理的，因此需要对其进行处理，减小处理的复杂度
            rois, roi_scores = postprocess_rpn_proposals(rpn_bbox_pred=rpn_box_pred,
                                                         rpn_cls_prob=rpn_cls_prob,
                                                         img_shape=img_shape,
                                                         anchors=anchors,
                                                         is_training=self.is_training)
            # rois shape [-1, 4]
            # +++++++++++++++++++++++++++++++++++++add img smry+++++++++++++++++++++++++++++++++++++++++++++++++++++++

            if self.is_training:
                rois_in_img = show_box_in_tensor.draw_boxes_with_scores(img_batch=input_img_batch,
                                                                        boxes=rois,
                                                                        scores=roi_scores)
                tf.summary.image('all_rpn_rois', rois_in_img)

                score_gre_05 = tf.reshape(tf.where(tf.greater_equal(roi_scores, 0.5)), [-1])
                score_gre_05_rois = tf.gather(rois, score_gre_05)
                score_gre_05_score = tf.gather(roi_scores, score_gre_05)
                score_gre_05_in_img = show_box_in_tensor.draw_boxes_with_scores(img_batch=input_img_batch,
                                                                                boxes=score_gre_05_rois,
                                                                                scores=score_gre_05_score)
                tf.summary.image('score_greater_05_rois', score_gre_05_in_img)
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # clw note：在模型训练过程中，需要对RPN网络不断的进行训练，因此以下则是在训练中针对于anchor产生层的训练设计方法：
        # 从所有的anchor中取一部分，计算其与ground truth之间的准确性
        # 计算之前得出的Anchors与ground truth的重叠率
        if self.is_training:
            with tf.variable_scope('sample_anchors_minibatch'):
                # clw note：这里会调用anchor_target_layer()这个函数
                #           tf.py_func的核心是一个func函数(由用户自己定义)，该函数接收numpy array作为输入，
                #           并返回numpy array类型的输出。看到这里，大家应该能够明白为什么建议使用py_func，
                #           因为在func函数中，可以对转化成numpy array的tensor进行np.运算，这就大大扩展了程序的灵活性。
                #           详见https://blog.csdn.net/tiankongtiankong01/article/details/80568311
                rpn_labels, rpn_bbox_targets = tf.py_func(anchor_target_layer,
                                                          [gtboxes_batch, img_shape, anchors],
                                                          [tf.float32, tf.float32])
                rpn_bbox_targets = tf.reshape(rpn_bbox_targets, [-1, 4])
                rpn_labels = tf.to_int32(rpn_labels, name="to_int32")
                rpn_labels = tf.reshape(rpn_labels, [-1])
                self.add_anchor_img_smry(input_img_batch, anchors, rpn_labels)

            # --------------------------------------add smry----------------------------------------------------------------

            # clw note：计算RPN分类的准确度，这里主要针对的是RPN网络是否预测出了尽可能多正确的背景框以及含有目标的框，
            # 这里不考虑那些label为-1 的框，只考虑 label为0 或者label为1的框，判断其准确度
            rpn_cls_category = tf.argmax(rpn_cls_prob, axis=1)
            kept_rpppn = tf.reshape(tf.where(tf.not_equal(rpn_labels, -1)), [-1])
            rpn_cls_category = tf.gather(rpn_cls_category, kept_rpppn)
            acc = tf.reduce_mean(tf.to_float(tf.equal(rpn_cls_category, tf.to_int64(tf.gather(rpn_labels, kept_rpppn)))))
            tf.summary.scalar('ACC/rpn_accuracy', acc)

            with tf.control_dependencies([rpn_labels]):
                # 刚才的设计都是针对于RPN网络的，并没有设计到真正的类别信息，以下则采用RCNN部分获得其相关的roi，target等信息句
                with tf.variable_scope('sample_RCNN_minibatch'):
                    rois, labels, bbox_targets = \
                    tf.py_func(proposal_target_layer,
                               [rois, gtboxes_batch],
                               [tf.float32, tf.float32, tf.float32])
                    rois = tf.reshape(rois, [-1, 4])
                    labels = tf.to_int32(labels)
                    labels = tf.reshape(labels, [-1])
                    bbox_targets = tf.reshape(bbox_targets, [-1, 4*(cfgs.CLASS_NUM+1)])
                    self.add_roi_batch_img_smry(input_img_batch, rois, labels)

        # -------------------------------------------------------------------------------------------------------------#
        #                                            Fast-RCNN                                                         #
        # -------------------------------------------------------------------------------------------------------------#

        # 5. build Fast-RCNN
        # rois = tf.Print(rois, [tf.shape(rois)], 'rois shape', summarize=10)
        bbox_pred, cls_score = self.build_fastrcnn(feature_to_cropped=feature_to_cropped, rois=rois, img_shape=img_shape)
        # bbox_pred shape: [-1, 4*(cls_num+1)].
        # cls_score shape： [-1, cls_num+1]

        cls_prob = slim.softmax(cls_score, 'cls_prob')


        # ----------------------------------------------add smry-------------------------------------------------------
        if self.is_training:
            cls_category = tf.argmax(cls_prob, axis=1)
            fast_acc = tf.reduce_mean(tf.to_float(tf.equal(cls_category, tf.to_int64(labels))))
            tf.summary.scalar('ACC/fast_acc', fast_acc)

        #  6. postprocess_fastrcnn
        # clw note：如果不进行训练，则此时直接进行解码
        # 就可以得到最终的bbox，每一个对应的分值，以及相关的box对应的类别信息。
        if not self.is_training:
            return self.postprocess_fastrcnn(rois=rois, bbox_ppred=bbox_pred, scores=cls_prob, img_shape=img_shape)
        # 否则，如果需要进行训练，则我们还需要计算出loss，进而采用优化方法来降低loss
        else:
            '''
            when trian. We need build Loss
            '''

            # 之前的Fast-RCNN网络引入了multi-task loss，在网络采用了全连接网络作为分类和边框回归，
            # 因此只有上图中第二个slice的Loss函数，在Faster-RCNN网络中，引入了RPN网络，在训练RPN网络时候，
            # 则引入了RPN网络的Loss函数，如上图中第一个slice。
            # clw note：原文：https://blog.csdn.net/zhao347316568/article/details/85028216

            # 有了如上的概念后，根据我们的网络，我们需要获取的参量为：
            # RPN网络：
            # 1.anchor的类别预测 pi 
            # 2.ground truth的类别标签 pi* 
            # 3. 256个位置对应的 256×（scale×ratios） 个anchor对应的编码后的 t 预测矩阵，
            # 4.每一个anchor对应的最大重叠率的ground truth bbox的 t target 矩阵

            # Fast-RCNN网络：
            # 1.真实的类别标签 u
            # 2.预测的类概率 p
            # 3.真实的ground truth 对应的 t 矩阵
            # 4.预测的bbox 对应的 t 矩阵

            # 在代码中，定义了一个loss_dict,用来保存RPN网络和Faster-RCNN的loss，其接收的参量如代码所示。
            # 进build_loss看一下：
            loss_dict = self.build_loss(rpn_box_pred=rpn_box_pred,
                                        rpn_bbox_targets=rpn_bbox_targets,
                                        rpn_cls_score=rpn_cls_score,
                                        rpn_labels=rpn_labels,
                                        bbox_pred=bbox_pred,
                                        bbox_targets=bbox_targets,
                                        cls_score=cls_score,
                                        labels=labels)


            final_bbox, final_scores, final_category = self.postprocess_fastrcnn(rois=rois,
                                                                                 bbox_ppred=bbox_pred,
                                                                                 scores=cls_prob,
                                                                                 img_shape=img_shape)
            return final_bbox, final_scores, final_category, loss_dict

    def get_restorer(self):
        checkpoint_path = tf.train.latest_checkpoint(os.path.join(cfgs.TRAINED_CKPT, cfgs.VERSION))

        if checkpoint_path != None:
            if cfgs.RESTORE_FROM_RPN:
                print('___restore from rpn___')
                model_variables = slim.get_model_variables()
                restore_variables = [var for var in model_variables if not var.name.startswith('FastRCNN_Head')] + \
                                    [slim.get_or_create_global_step()]
                for var in restore_variables:
                    print(var.name)
                restorer = tf.train.Saver(restore_variables)
            else:
                restorer = tf.train.Saver()
            print("model restore from :", checkpoint_path)
        else:
            checkpoint_path = cfgs.PRETRAINED_CKPT
            print("model restore from pretrained mode, path is :", checkpoint_path)

            model_variables = slim.get_model_variables()
            # for var in model_variables:
            #     print(var.name)
            # print(20*"__++__++__")

            def name_in_ckpt_rpn(var):
                return var.op.name

            def name_in_ckpt_fastrcnn_head(var):
                '''
                Fast-RCNN/resnet_v1_50/block4 -->resnet_v1_50/block4
                Fast-RCNN/MobilenetV2/** -- > MobilenetV2 **
                :param var:
                :return:
                '''
                return '/'.join(var.op.name.split('/')[1:])

            nameInCkpt_Var_dict = {}
            for var in model_variables:
                if var.name.startswith('Fast-RCNN/'+self.base_network_name):  # +'/block4'
                    var_name_in_ckpt = name_in_ckpt_fastrcnn_head(var)
                    nameInCkpt_Var_dict[var_name_in_ckpt] = var
                else:
                    if var.name.startswith(self.base_network_name):
                        var_name_in_ckpt = name_in_ckpt_rpn(var)
                        nameInCkpt_Var_dict[var_name_in_ckpt] = var
                    else:
                        continue
            restore_variables = nameInCkpt_Var_dict
            for key, item in restore_variables.items():
                print("var_in_graph: ", item.name)
                print("var_in_ckpt: ", key)
                print(20*"___")
            restorer = tf.train.Saver(restore_variables)
            print(20 * "****")
            print("restore from pretrained_weighs in IMAGE_NET")
        return restorer, checkpoint_path

    def get_gradients(self, optimizer, loss):
        '''

        :param optimizer:
        :param loss:
        :return:

        return vars and grads that not be fixed
        '''

        # if cfgs.FIXED_BLOCKS > 0:
        #     trainable_vars = tf.trainable_variables()
        #     # trained_vars = slim.get_trainable_variables()
        #     start_names = [cfgs.NET_NAME + '/block%d'%i for i in range(1, cfgs.FIXED_BLOCKS+1)] + \
        #                   [cfgs.NET_NAME + '/conv1']
        #     start_names = tuple(start_names)
        #     trained_var_list = []
        #     for var in trainable_vars:
        #         if not var.name.startswith(start_names):
        #             trained_var_list.append(var)
        #     # slim.learning.train()
        #     grads = optimizer.compute_gradients(loss, var_list=trained_var_list)
        #     return grads
        # else:
        #     return optimizer.compute_gradients(loss)
        return optimizer.compute_gradients(loss)

    def enlarge_gradients_for_bias(self, gradients):

        final_gradients = []
        with tf.variable_scope("Gradient_Mult") as scope:
            for grad, var in gradients:
                scale = 1.0
                if cfgs.MUTILPY_BIAS_GRADIENT and './biases' in var.name:
                    scale = scale * cfgs.MUTILPY_BIAS_GRADIENT
                if not np.allclose(scale, 1.0):
                    grad = tf.multiply(grad, scale)
                final_gradients.append((grad, var))
        return final_gradients




















