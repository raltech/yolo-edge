#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : yolov3.py
#   Author      : YunYang1994
#   Created date: 2019-07-12 13:47:10
#   Description :
#
#================================================================

import numpy as np
import tensorflow as tf
import core.utils as utils
import core.common as common
import core.backbone as backbone
from core.config import cfg


NUM_CLASS       = len(utils.read_class_names(cfg.YOLO.CLASSES))
ANCHORS         = utils.get_anchors(cfg.YOLO.ANCHORS)
STRIDES         = np.array(cfg.YOLO.STRIDES)
IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH

def YOLOv3(input_layer):
    route_1, route_2, conv = backbone.darknet53(input_layer)

    conv = common.convolutional(conv, (1, 1, 1024,  512))
    conv = common.convolutional(conv, (3, 3,  512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024,  512))
    conv = common.convolutional(conv, (3, 3,  512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024,  512))

    conv_lobj_branch = common.convolutional(conv, (3, 3, 512, 1024))
    conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 1024, 3*(NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(conv, (1, 1,  512,  256))
    conv = common.upsample(conv)

    conv = tf.concat([conv, route_2], axis=-1)

    conv = common.convolutional(conv, (1, 1, 768, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))

    conv_mobj_branch = common.convolutional(conv, (3, 3, 256, 512))
    conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 512, 3*(NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.upsample(conv)

    conv = tf.concat([conv, route_1], axis=-1)

    conv = common.convolutional(conv, (1, 1, 384, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))

    conv_sobj_branch = common.convolutional(conv, (3, 3, 128, 256))
    conv_sbbox = common.convolutional(conv_sobj_branch, (1, 1, 256, 3*(NUM_CLASS +5)), activate=False, bn=False)

    return [conv_sbbox, conv_mbbox, conv_lbbox]

def decode(conv_output, i=0):
    """
    return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
    => [batch_size, output_size, output_size, anchor_per_scale * (5 + num_classes)]
            contains (x, y, w, h, score, probability)
    i: size level (0:small, 1:medium, 2:large)

    this function decode raw convolution output from YOLOv3 into processed output
    """
    # print(['conv_output', conv_output.shape])
    # =>  ['conv_output', TensorShape([None, 52, 52, 255])]
    #     ['conv_output', TensorShape([None, 26, 26, 255])]
    #     ['conv_output', TensorShape([None, 13, 13, 255])]
    conv_shape       = tf.shape(conv_output)
    batch_size       = conv_shape[0] # e.g., 4
    output_size      = conv_shape[1] # i.e., 52 for small, 26 for medium, 13 for large

    # Let's use conv_out as 4-d as it is 
    conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
    conv_raw_conf = conv_output[:, :, :, :, 4:5]
    conv_raw_prob = conv_output[:, :, :, :, 5: ]

    y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size]) # 52x52, 26x26, 13x13
    '''
    <tf.Tensor: id=47, shape=(13, 13), dtype=int32, numpy=
    array([ [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
            [ 2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
            [ 3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3],
            [ 4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4],
            [ 5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5],
            [ 6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6],
            [ 7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7],
            [ 8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8],
            [ 9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9],
            [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
            [11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11],
            [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]], dtype=int32)>
    '''
    x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1]) # 52x52,...
    '''
    <tf.Tensor: id=57, shape=(13, 13), dtype=int32, numpy=
    array([ [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
            [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
            [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
            [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
            [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
            [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
            [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
            [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
            [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
            [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
            [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
            [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
            [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12]], dtype=int32)>
    '''

    xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1) # simply stack them on top
    xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)
    # => TensorShape([batch_size(=4), 13, 13, 3, 2])

    # Scale as described in the original paper
    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * STRIDES[i]
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i]) * STRIDES[i]
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
    '''
    print('decode')
    print(pred_xy.shape)   => (None, None, None, 3, 2)
    print(pred_wh.shape)   => (None, None, None, 3, 2)
    print(pred_xywh.shape) => (None, None, None, 3, 4)
    so, last 4 for x,y,w,h
    '''

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)
    '''
    print('decode')
    print(pred_conf.shape) => (None, None, None, 3, 1)
    print(pred_prob.shape) => (None, None, None, 3, 80)
 
    print(tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1).shape)
    => (None, None, None, 3, 85)
    '''

    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

def bbox_iou(boxes1, boxes2):
    # print('core/yolov3.py/bbox_iou')
    '''
    small bbox
    boxes1.shape:   (4, 52, 52, 3, 1, 4) 
    boxes2.shape:   (4, 1, 1, 1, 150, 4)

    medium bbox
    boxes1.shape:   (4, 26, 26, 3, 1, 4)
    boxes2.shape:   (4, 1, 1, 1, 150, 4)
    
    large bbox
    boxes1.shape:   (4, 13, 13, 3, 1, 4)
    boxes2.shape:   (4, 1, 1, 1, 150, 4)

    4 ??
    52x52 cells
    3 different anchors for the level
    4 for x,y,w,h

    '''
    
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    return 1.0 * inter_area / union_area

def bbox_giou(boxes1, boxes2):

    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / union_area

    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]
    giou = iou - 1.0 * (enclose_area - union_area) / enclose_area
    # print('giou')
    # print(giou.shape)

    return giou


def compute_loss(pred, conv, label, bboxes, i=0):
    '''
    i argument specify the scaling level (0=small, 1=medium, 2=large)

    From train.py at line 95
    batch_label_sbbox => label
    batch_sbboxes => bboxes
    
    print('compute_loss')
    print(['pred', pred.shape])
    print(['conv', conv.shape])
    print(['label,',label.shape])
    print(['bboxes',bboxes.shape])
    ------------
    compute_loss (small scale)
    ['pred', TensorShape([4, 52, 52, 3, 85])]
    ['conv', TensorShape([4, 52, 52, 255])]
    ['label,', (4, 52, 52, 3, 85)]
    ['bboxes', (4, 150, 4)]
    compute_loss (medium scale)
    ['pred', TensorShape([4, 26, 26, 3, 85])]
    ['conv', TensorShape([4, 26, 26, 255])]
    ['label,', (4, 26, 26, 3, 85)]
    ['bboxes', (4, 150, 4)]
    compute_loss (large scale)
    ['pred', TensorShape([4, 13, 13, 3, 85])]
    ['conv', TensorShape([4, 13, 13, 255])]
    ['label,', (4, 13, 13, 3, 85)]
    ['bboxes', (4, 150, 4)]
    '''

    conv_shape  = tf.shape(conv)
    batch_size  = conv_shape[0]
    output_size = conv_shape[1]
    input_size  = STRIDES[i] * output_size
    conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_conf = conv[:, :, :, :, 4:5]
    conv_raw_prob = conv[:, :, :, :, 5:]

    pred_xywh     = pred[:, :, :, :, 0:4]
    pred_conf     = pred[:, :, :, :, 4:5] # pred_conf = sigmoid(conv_raw_conf)

    label_xywh    = label[:, :, :, :, 0:4]
    respond_bbox  = label[:, :, :, :, 4:5] # respond is same as ground truth confidence (so 1 or 0 ??)
    label_prob    = label[:, :, :, :, 5:]

    giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)
    input_size = tf.cast(input_size, tf.float32)

    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    giou_loss = respond_bbox * bbox_loss_scale * (1- giou)

    # print('Input to bbox_iou() func')
    # print(['pred_xywh[:, :, :, :, np.newaxis, :]',pred_xywh[:, :, :, :, np.newaxis, :].shape])
    # => ['pred_xywh[:, :, :, :, np.newaxis, :]', TensorShape([4, 52, 52, 3, 1, 4])]
    # print(['bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :]',bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :].shape])
    # => ['bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :]', (4, 1, 1, 1, 150, 4)]
    iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
    # print(['iou', iou.shape])
    # ['iou', TensorShape([4, 52, 52, 3, 150])] i = 0
    # ['iou', TensorShape([4, 26, 26, 3, 150])] i = 1
    # ['iou', TensorShape([4, 13, 13, 3, 150])] i = 2

    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1) # Choose one bbox that has max iou from 150 bboxes
    # print(['max_out', max_iou.shape])
    # ['max_out', TensorShape([4, 52, 52, 3, 1])] i = 0 
    # ['max_out', TensorShape([4, 26, 26, 3, 1])] i = 1
    # ['max_out', TensorShape([4, 13, 13, 3, 1])] i = 2

    respond_bgd = (1.0 - respond_bbox) * tf.cast( max_iou < IOU_LOSS_THRESH, tf.float32 )

    conf_focal = tf.pow(respond_bbox - pred_conf, 2)

    conf_loss = conf_focal * (
            respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
            +
            respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
    )

    # print([respond_bbox.shape, label_prob.shape, conv_raw_prob.shape])
    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

    # print(['giou_loss', giou_loss.shape]) => ['giou_loss', TensorShape([4, 52, 52, 3, 1])]
    # print(['conf_loss', conf_loss.shape]) => ['conf_loss', TensorShape([4, 52, 52, 3, 1])]
    # print(['prob_loss', prob_loss.shape]) => ['prob_loss', TensorShape([4, 52, 52, 3, 80])]
    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4])) # reduce_sum reduces into (4,) tensor
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4])) # then take mean of those 4 values
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4])) # thus, final loss is scalar
    # print(['giou_loss', giou_loss]) 
    # => ['giou_loss', <tf.Tensor: id=16226, shape=(), dtype=float32, numpy=2.2939475>]
    # print(['conf_loss', conf_loss])
    # => ['conf_loss', <tf.Tensor: id=16230, shape=(), dtype=float32, numpy=1525.9078>]
    # print(['prob_loss', prob_loss])
    # => ['prob_loss', <tf.Tensor: id=16234, shape=(), dtype=float32, numpy=96.9215>]

    return giou_loss, conf_loss, prob_loss





