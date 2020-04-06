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

def decode(conv_output, i=0): # Edge Compatible
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
    # conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))


    # conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
    conv_raw_dxdy = tf.concat((conv_output[:,:,:,0:2], conv_output[:,:,:,85:87], conv_output[:,:,:,170:172]), -1)
    # conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
    conv_raw_dwdh = tf.concat((conv_output[:,:,:,2:4], conv_output[:,:,:,87:89], conv_output[:,:,:,172:174]), -1)
    # conv_raw_conf = conv_output[:, :, :, :, 4:5]
    conv_raw_conf = tf.concat((conv_output[:,:,:,4:5], conv_output[:,:,:,89:90], conv_output[:,:,:,174:175]), -1)
    # conv_raw_prob = conv_output[:, :, :, :, 5: ]
    conv_raw_prob = tf.concat((conv_output[:,:,:,5:85], conv_output[:,:,:,90:170], conv_output[:,:,:,175:255]), -1)

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
    # => 13x13x2
    # xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1])
    xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, :], [batch_size, 1, 1, 3]) # => 4x13x13x6
    xy_grid = tf.cast(xy_grid, tf.float32)
    # => TensorShape([batch_size(=4), 13, 13, 3, 2]) (original)
    # => 4,13,13,6 (modified)

    # Scale as described in the original paper
    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * STRIDES[i]
    pred_wh = (tf.exp(conv_raw_dwdh) * tf.reshape(ANCHORS[i], [-1])) * STRIDES[i]
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
    '''
    print('decode')
    print(pred_xy.shape)   => (None, None, None, 3, 2)
    print(pred_wh.shape)   => (None, None, None, 3, 2)
    print(pred_xywh.shape) => (None, None, None, 3, 4)
    so, last 4 for x,y,w,h
    '''

    # print('decode')
    # print(pred_xywh.shape) => (None, 52, 52, 12)

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
    # Compute giou per anchor scale (so iterate 3 times for small, med, large)

    # boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
    #                     boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    # boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
    #                     boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)
    # boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]), # :2 => - part, 2: => + part
                        # tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    # boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
    #                     tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_original = tf.identity(boxes1)
    boxes2_original = tf.identity(boxes2)

    ### Small level ###
    boxes1 = tf.concat([boxes1_original[..., 0:2] - boxes1_original[..., 2:4] * 0.5,
                        boxes1_original[..., 0:2] + boxes1_original[..., 2:4] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2_original[..., 0:2] - boxes2_original[..., 2:4] * 0.5,
                        boxes2_original[..., 0:2] + boxes2_original[..., 2:4] * 0.5], axis=-1)
    boxes1 = tf.concat([tf.minimum(boxes1[..., 0:2], boxes1[..., 2:4]), 
                        tf.maximum(boxes1[..., 0:2], boxes1[..., 2:4])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., 0:2], boxes2[..., 2:4]), 
                        tf.maximum(boxes2[..., 0:2], boxes2[..., 2:4])], axis=-1)

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
    giou_small = iou - 1.0 * (enclose_area - union_area) / enclose_area 

    ### Medium level ###
    # print(boxes1_original.shape)
    boxes1 = tf.concat([boxes1_original[..., 4:6] - boxes1_original[..., 6:8] * 0.5,
                        boxes1_original[..., 4:6] + boxes1_original[..., 6:8] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2_original[..., 4:6] - boxes2_original[..., 6:8] * 0.5,
                        boxes2_original[..., 4:6] + boxes2_original[..., 6:8] * 0.5], axis=-1)
    boxes1 = tf.concat([tf.minimum(boxes1[..., 0:2], boxes1[..., 2:4]), 
                        tf.maximum(boxes1[..., 0:2], boxes1[..., 2:4])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., 0:2], boxes2[..., 2:4]), 
                        tf.maximum(boxes2[..., 0:2], boxes2[..., 2:4])], axis=-1)

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
    giou_medium = iou - 1.0 * (enclose_area - union_area) / enclose_area

    ### Large level ###
    boxes1 = tf.concat([boxes1_original[..., 8:10] - boxes1_original[..., 10:12] * 0.5,
                        boxes1_original[..., 8:10] + boxes1_original[..., 10:12] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2_original[..., 8:10] - boxes2_original[..., 10:12] * 0.5,
                        boxes2_original[..., 8:10] + boxes2_original[..., 10:12] * 0.5], axis=-1)
    boxes1 = tf.concat([tf.minimum(boxes1[..., 0:2], boxes1[..., 2:4]), 
                        tf.maximum(boxes1[..., 0:2], boxes1[..., 2:4])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., 0:2], boxes2[..., 2:4]), 
                        tf.maximum(boxes2[..., 0:2], boxes2[..., 2:4])], axis=-1)

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
    giou_large = iou - 1.0 * (enclose_area - union_area) / enclose_area
    
    return tf.concat([giou_small[...,tf.newaxis], giou_medium[...,tf.newaxis], giou_large[...,tf.newaxis]], axis=-1)
    # giou.shape => (4, 52, 52, 3)


    ''' attemp 1
    boxes1 = tf.concat(
             (tf.concat((boxes1[..., 0:2], boxes1[..., 4:6], boxes1[..., 8:10]), axis=-1) - \
              tf.concat((boxes1[..., 2:4], boxes1[..., 6:8], boxes1[..., 10:12]), axis=-1) * 0.5),
             (tf.concat((boxes1[..., 0:2], boxes1[..., 4:6], boxes1[..., 8:10]), axis=-1) + \
              tf.concat((boxes1[..., 2:4], boxes1[..., 6:8], boxes1[..., 10:12]), axis=-1) * 0.5), axis=-1)
    
    boxes2 = tf.concat(
             (tf.concat((boxes2[..., 0:2], boxes2[..., 4:6], boxes2[..., 8:10]), axis=-1) - \
              tf.concat((boxes2[..., 2:4], boxes2[..., 6:8], boxes2[..., 10:12]), axis=-1) * 0.5),
             (tf.concat((boxes2[..., 0:2], boxes2[..., 4:6], boxes2[..., 8:10]), axis=-1) + \
              tf.concat((boxes2[..., 2:4], boxes2[..., 6:8], boxes2[..., 10:12]), axis=-1) * 0.5), axis=-1)

    
    boxes1 = tf.concat(
            (tf.concat(tf.minimum(boxes1[..., 0:2], boxes1[..., 2:4]), 
                       tf.minimum(boxes1[..., 4:6], boxes1[..., 6:8]), 
                       tf.minimum(boxes1[..., 8:10], boxes1[..., 10:12]), axis=-1)),
            (tf.concat(tf.maximum(boxes1[..., 0:2], boxes1[..., 2:4]), 
                       tf.maximum(boxes1[..., 4:6], boxes1[..., 6:8]), 
                       tf.maximum(boxes1[..., 8:10], boxes1[..., 10:12]), axis=-1)), axis=-1)
    
    boxes2 = tf.concat(
            (tf.concat(tf.minimum(boxes2[..., 0:2], boxes2[..., 2:4]), 
                       tf.minimum(boxes2[..., 4:6], boxes2[..., 6:8]), 
                       tf.minimum(boxes2[..., 8:10], boxes2[..., 10:12]), axis=-1)),
            (tf.concat(tf.maximum(boxes2[..., 0:2], boxes2[..., 2:4]), 
                       tf.maximum(boxes2[..., 4:6], boxes2[..., 6:8]), 
                       tf.maximum(boxes2[..., 8:10], boxes2[..., 10:12]), axis=-1)), axis=-1)
    '''

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
    # conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    # conv_raw_conf = conv[:, :, :, :, 4:5]
    conv_raw_conf = tf.concat((conv[:,:,:,4:5], conv[:,:,:,89:90], conv[:,:,:,174:175]), -1) 
    # conv_raw_prob = conv[:, :, :, :, 5:]
    conv_raw_prob = tf.concat((conv[:,:,:,5:85], conv[:,:,:,90:170], conv[:,:,:,175:255]), -1) 

    # pred_xywh     = pred[:, :, :, :, 0:4]
    pred_xywh     = tf.concat((pred[:,:,:,0:4], pred[:,:,:,85:89], pred[:,:,:,170:174]), -1) 
    # pred_conf     = pred[:, :, :, :, 4:5] # pred_conf = sigmoid(conv_raw_conf)
    pred_conf     = tf.concat((pred[:,:,:,4:5], pred[:,:,:,89:90], pred[:,:,:,174:175]), -1) 

    # label_xywh    = label[:, :, :, :, 0:4]
    label_xywh    = tf.concat((label[:,:,:,0:4], label[:,:,:,85:89], label[:,:,:,170:174]), -1)
    # respond_bbox  = label[:, :, :, :, 4:5] # respond is same as ground truth confidence (so 1 or 0 ??)
    respond_bbox  = tf.concat((label[:,:,:,4:5], label[:,:,:,89:90], label[:,:,:,174:175]), -1)
    # label_prob    = label[:, :, :, :, 5:]
    label_prob    = tf.concat((label[:,:,:,5:85], label[:,:,:,90:170], label[:,:,:,175:255]), -1)

    # giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1) # why expand...????
    giou = bbox_giou(pred_xywh, label_xywh)
    # print('green')
    # print(giou.shape) => (4, 52, 52, 3)
    input_size = tf.cast(input_size, tf.float32)

    # bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    bbox_loss_scale = 2.0 - 1.0 * (tf.concat((label_xywh[:, :, :, 2:3], label_xywh[:, :, :, 87:88], label_xywh[:, :, :, 172:173]), -1)) * \
                                  (tf.concat((label_xywh[:, :, :, 3:4], label_xywh[:, :, :, 88:89], label_xywh[:, :, :, 173:174]), -1)) / \
                                  (input_size ** 2)
    # print(respond_bbox.shape) => (4, 52, 52, 3)
    # print(bbox_loss_scale.shape) => (4, 52, 52, 1)
    # print(giou.shape) => (4, 52, 52, 3)
    giou_loss = respond_bbox * bbox_loss_scale * (1- giou)
    # print(giou_loss.shape) => (4, 52, 52, 3)

    # print('Input to bbox_iou() func')
    # print(['pred_xywh[:, :, :, :, np.newaxis, :]',pred_xywh[:, :, :, :, np.newaxis, :].shape])
    # => ['pred_xywh[:, :, :, :, np.newaxis, :]', TensorShape([4, 52, 52, 3, 1, 4])]
    # print(['bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :]',bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :].shape])
    # => ['bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :]', (4, 1, 1, 1, 150, 4)]

    # print(pred_xywh.shape) => (4, 52, 52, 12)
    # print(bboxes.shape) => (4, 150, 4)
    # idea: make bboxes.shape into (4,150,12)
    # for each bbox from 150, pass to bbox_iou() to calc iou
    # then marge 150 ious altogether
    # bboxes = tf.concat((bboxes, bboxes, bboxes), axis=-1)

    pred_xywh_small = pred_xywh[..., 0:4] 
    pred_xywh_medium = pred_xywh[..., 4:8]
    pred_xywh_large = pred_xywh[..., 8:12] # (4, 52, 52, 4)
    max_bbox_per_scale = bboxes.shape[1]
    # iou_small = tf.zeros([pred_xywh_small.shape[0], pred_xywh_small.shape[1], pred_xywh_small.shape[2], max_bbox_per_scale])
    for i in range(max_bbox_per_scale):
        # print(pred_xywh_small.shape) => (4, 52, 52, 4)
        # print(bboxes[:,i:i+1,tf.newaxis,:].shape) => (4, 1, 1, 4)
        if i == 0:
            iou_small  = bbox_iou(pred_xywh_small, bboxes[:,i:i+1,tf.newaxis,:])[...,tf.newaxis] 
            iou_medium = bbox_iou(pred_xywh_medium, bboxes[:,i:i+1,tf.newaxis,:])[...,tf.newaxis]  
            iou_large  = bbox_iou(pred_xywh_large, bboxes[:,i:i+1,tf.newaxis,:])[...,tf.newaxis]   
        else:
            iou_small  = tf.concat((iou_small, bbox_iou(pred_xywh_small, bboxes[:,i:i+1,tf.newaxis,:])[...,tf.newaxis]), -1)   
            iou_medium  = tf.concat((iou_medium, bbox_iou(pred_xywh_medium, bboxes[:,i:i+1,tf.newaxis,:])[...,tf.newaxis]), -1)   
            iou_large  = tf.concat((iou_large, bbox_iou(pred_xywh_large, bboxes[:,i:i+1,tf.newaxis,:])[...,tf.newaxis]), -1)
    # print(iou_small.shape) => (4, 52, 52, 150)

    # iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
    # print(['iou', iou.shape])
    # ['iou', TensorShape([4, 52, 52, 3, 150])] i = 0
    # ['iou', TensorShape([4, 26, 26, 3, 150])] i = 1
    # ['iou', TensorShape([4, 13, 13, 3, 150])] i = 2

    max_iou_small = tf.reduce_max(iou_small, axis=-1)   # Choose one bbox that has max iou from 150 bboxes
    max_iou_medium = tf.reduce_max(iou_medium, axis=-1) # Choose one bbox that has max iou from 150 bboxes
    max_iou_large = tf.reduce_max(iou_large, axis=-1)   # Choose one bbox that has max iou from 150 bboxes
    # print(max_iou_small.shape) => (4, 52, 52)
    max_iou = tf.concat((max_iou_small[...,tf.newaxis], max_iou_medium[...,tf.newaxis], max_iou_large[...,tf.newaxis]), -1)
    # print(max_iou.shape) => (4, 52, 52, 3)

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
    # => original: [(4, 13, 13, 3, 1), (4, 13, 13, 3, 80), TensorShape([4, 13, 13, 3, 80])]
    # => [TensorShape([4, 52, 52, 3]), TensorShape([4, 52, 52, 240]), TensorShape([4, 52, 52, 240])]
    # print(respond_bbox.numpy().shape) => (4, 52, 52, 3)
    # divide them into each anchor level so that the last dim of respond_bbox becomes 1 and 80
    respond_bbox_small  = respond_bbox[...,0,tf.newaxis]
    respond_bbox_medium = respond_bbox[...,1,tf.newaxis]
    respond_bbox_large  = respond_bbox[...,2,tf.newaxis]
    label_prob_small  = label_prob[...,0:80]
    label_prob_medium = label_prob[...,80:160]
    label_prob_large  = label_prob[...,160:240]
    conv_raw_prob_small  = conv_raw_prob[...,0:80]
    conv_raw_prob_medium = conv_raw_prob[...,80:160]
    conv_raw_prob_large  = conv_raw_prob[...,160:240]
    prob_loss_small = respond_bbox_small * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob_small, logits=conv_raw_prob_small)
    prob_loss_medium = respond_bbox_medium * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob_medium, logits=conv_raw_prob_medium)
    prob_loss_large = respond_bbox_large * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob_large, logits=conv_raw_prob_large)
    # print(prob_loss_small.shape) => (4, 52, 52, 80)
    # prob_loss = respond_bbox.numpy() * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)
    prob_loss = tf.concat((prob_loss_small, prob_loss_medium, prob_loss_large), -1)
    # print(prob_loss.shape) => (4, 52, 52, 240)

    # print(['giou_loss', giou_loss.shape]) => ['giou_loss', TensorShape([4, 52, 52, 3, 1])]
    # print(['conf_loss', conf_loss.shape]) => ['conf_loss', TensorShape([4, 52, 52, 3, 1])]
    # print(['prob_loss', prob_loss.shape]) => ['prob_loss', TensorShape([4, 52, 52, 3, 80])]
    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3])) # reduce_sum reduces into (4,) tensor
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3])) # then take mean of those 4 values
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3])) # thus, final loss is scalar
    # print(['giou_loss', giou_loss]) 
    # => ['giou_loss', <tf.Tensor: id=16226, shape=(), dtype=float32, numpy=2.2939475>]
    # print(['conf_loss', conf_loss])
    # => ['conf_loss', <tf.Tensor: id=16230, shape=(), dtype=float32, numpy=1525.9078>]
    # print(['prob_loss', prob_loss])
    # => ['prob_loss', <tf.Tensor: id=16234, shape=(), dtype=float32, numpy=96.9215>]
    
    return giou_loss, conf_loss, prob_loss





