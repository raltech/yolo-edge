#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : train.py
#   Author      : YunYang1994
#   Created date: 2019-07-18 09:18:54
#   Description :
#
#================================================================

import os
import time
import shutil
import numpy as np
import tensorflow as tf
import core.utils as utils
from tqdm import tqdm
from core.dataset import Dataset
from core.yolov3 import YOLOv3, decode, compute_loss
from core.config import cfg

trainset = Dataset('train')
logdir = "./data/log"
steps_per_epoch = len(trainset)
global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch
total_steps = cfg.TRAIN.EPOCHS * steps_per_epoch

input_tensor = tf.keras.layers.Input([416, 416, 3])
conv_tensors = YOLOv3(input_tensor)
'''
print('train.py')
print(len(conv_tensors)) => 3 
print(conv_tensors[0].shape) => (None, 52, 52, 255)
print(conv_tensors[1].shape) => (None, 26, 26, 255)
print(conv_tensors[2].shape) => (None, 13, 13, 255)
print(np.array(conv_tensors).shape) => (3,)

so...
conv_tensors: [(None, 52, 52, 255), (None, 26, 26, 255), (None, 13, 13, 255)]
'''

'''
pred_tensor is derived from conv_tensor which is a raw output from YOLO network
'''
output_tensors = []
'''note
i iterates 0, 1, 2
0 for small size anchor
1 for medium size anchor
2 for large size anchor
'''
for i, conv_tensor in enumerate(conv_tensors):
    pred_tensor = decode(conv_tensor, i)
    # print('make sure...')
    # print(conv_tensor.shape) => (None, 52, 52, 255)
    # print(pred_tensor.shape) => (None, 52, 52, 255)
    # Looks good!
    output_tensors.append(conv_tensor)
    output_tensors.append(pred_tensor)

model = tf.keras.Model(input_tensor, output_tensors)
optimizer = tf.keras.optimizers.Adam()
if os.path.exists(logdir): shutil.rmtree(logdir)
writer = tf.summary.create_file_writer(logdir)

def train_step(image_data, batch_label_sbbox, batch_sbboxes, batch_label_mbbox, batch_mbboxes, batch_label_lbbox, batch_lbboxes):
    with tf.GradientTape() as tape:
        pred_result = model(image_data, training=True)
        '''
        print('train.py/train_step')
        pred_result is list
        print(np.array(pred_result).shape) : (6, 4)
        print(pred_result[0].shape) : (4, 52, 52, 255)   => conv
        print(pred_result[1].shape) : (4, 52, 52, 3, 85) => pred
        print(pred_result[2].shape) : (4, 26, 26, 255)   => conv
        print(pred_result[3].shape) : (4, 26, 26, 3, 85) => pred
        print(pred_result[4].shape) : (4, 13, 13, 255)   => conv
        print(pred_result[5].shape) : (4, 13, 13, 3, 85) => pred
        '''
        giou_loss=conf_loss=prob_loss=0

        # optimizing process
        for i in range(3): # Iterating through each scale levels (small/medium/large)
            conv, pred = pred_result[i*2], pred_result[i*2+1]
            '''
            print('train.py/train_step')
            print(conv.shape) : (4, 52, 52, 255)
                                (4, 26, 26, 255)
                                (4, 13, 13, 255)
            print(pred.shape) : (4, 52, 52, 3, 85)
                                (4, 26, 26, 3, 85)
                                (4, 13, 13, 3, 85)
            '''
            # loss_items = compute_loss(pred, conv, *target[i], i)
            if i == 0:
                loss_items = compute_loss(pred, conv, batch_label_sbbox, batch_sbboxes, i)
            elif i == 1:
                loss_items = compute_loss(pred, conv, batch_label_mbbox, batch_mbboxes, i)
            else:
                loss_items = compute_loss(pred, conv, batch_label_lbbox, batch_lbboxes, i)
            # loss_items => [giou_loss, conf_loss, prob_loss]

            '''
            List of (batch_smaller_target, batch_medium_target, batch_larger_target)
            print(len(target)) => 3 | select (batch_smaller_target, batch_medium_target, batch_larger_target)
            print(len(target[0])) => 2 | select (batch_label_sbbox, batch_sbboxes)

            ----- (for batch_label_sbbox)
            print(len(target[0][0])) => 4 | batch size 
            print(len(target[0][0][0])) => 52 | cell size
            print(len(target[0][0][0][0])) => 52 | cell size
            print(len(target[0][0][0][0][0])) => 3 | 3 anchors per each level
            print(len(target[0][0][0][0][0][0])) => 85 | xywh(4), conf(1), prob(80)

            ----- (for batch_sbboxes)
            print(len(target[0][1]))    => 4   (Batch size)
            print(len(target[0][1][0])) => 150 (Each image contains maximum 150 bboxes)
            print(len(target[0][1][1])) => 150
            print(len(target[0][1][2])) => 150
            print(len(target[0][1][3])) => 150
            print(len(target[0][1][0][120])) => 4 (Each bbox has xywh info)
            '''
            
            giou_loss += loss_items[0]
            conf_loss += loss_items[1]
            prob_loss += loss_items[2]

        total_loss = giou_loss + conf_loss + prob_loss

        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        tf.print("=> STEP %4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                 "prob_loss: %4.2f   total_loss: %4.2f" %(global_steps, optimizer.lr.numpy(),
                                                          giou_loss, conf_loss,
                                                          prob_loss, total_loss))
        # update learning rate
        global_steps.assign_add(1)
        if global_steps < warmup_steps:
            lr = global_steps / warmup_steps *cfg.TRAIN.LR_INIT
        else:
            lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
                (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
            )
        optimizer.lr.assign(lr.numpy())

        # writing summary data
        with writer.as_default():
            tf.summary.scalar("lr", optimizer.lr, step=global_steps)
            tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
            tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
            tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
            tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
        writer.flush()


for epoch in range(cfg.TRAIN.EPOCHS):
    for image_data, batch_label_sbbox, batch_sbboxes, batch_label_mbbox, batch_mbboxes, batch_label_lbbox, batch_lbboxes in trainset:
        '''
        image_data = batch_image
        target = (batch_smaller_target, batch_medium_target, batch_larger_target)
            where,
                batch_smaller_target = batch_label_sbbox, batch_sbboxes
                batch_medium_target  = batch_label_mbbox, batch_mbboxes
                batch_larger_target  = batch_label_lbbox, batch_lbboxes
        '''
        train_step(image_data, batch_label_sbbox, batch_sbboxes, batch_label_mbbox, batch_mbboxes, batch_label_lbbox, batch_lbboxes)
    model.save_weights("./yolov3")

