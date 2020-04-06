#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : dataset.py
#   Author      : YunYang1994
#   Created date: 2019-03-15 18:05:03
#   Description :
#
#================================================================

import os
import cv2
import random
import numpy as np
import tensorflow as tf
import core.utils as utils
from core.config import cfg



class Dataset(object):
    """implement Dataset here"""
    def __init__(self, dataset_type):
        self.annot_path  = cfg.TRAIN.ANNOT_PATH if dataset_type == 'train' else cfg.TEST.ANNOT_PATH
        self.input_sizes = cfg.TRAIN.INPUT_SIZE if dataset_type == 'train' else cfg.TEST.INPUT_SIZE
        self.batch_size  = cfg.TRAIN.BATCH_SIZE if dataset_type == 'train' else cfg.TEST.BATCH_SIZE
        self.data_aug    = cfg.TRAIN.DATA_AUG   if dataset_type == 'train' else cfg.TEST.DATA_AUG

        self.train_input_sizes = cfg.TRAIN.INPUT_SIZE
        self.strides = np.array(cfg.YOLO.STRIDES)
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.anchors = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.max_bbox_per_scale = 150

        self.annotations = self.load_annotations(dataset_type)
        self.num_samples = len(self.annotations)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0


    def load_annotations(self, dataset_type):
        with open(self.annot_path, 'r') as f:
            txt = f.readlines()
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
        np.random.shuffle(annotations)
        return annotations

    def __iter__(self):
        return self

    def __next__(self):

        # with tf.device('/cpu:0'):
        self.train_input_size = random.choice(self.train_input_sizes)
        self.train_output_sizes = self.train_input_size // self.strides
        # print('__next__')
        # print(self.train_input_size) => 416 (COCO's image size)
        # print(self.train_output_sizes) => [52 26 13] (grid size for small, medium, large anchors)

        batch_image = np.zeros((self.batch_size, self.train_input_size, self.train_input_size, 3), dtype=np.float32)

        batch_label_sbbox = np.zeros((self.batch_size, self.train_output_sizes[0], self.train_output_sizes[0],
                                        self.anchor_per_scale * (5 + self.num_classes)), dtype=np.float32)
        batch_label_mbbox = np.zeros((self.batch_size, self.train_output_sizes[1], self.train_output_sizes[1],
                                        self.anchor_per_scale * (5 + self.num_classes)), dtype=np.float32)
        batch_label_lbbox = np.zeros((self.batch_size, self.train_output_sizes[2], self.train_output_sizes[2],
                                        self.anchor_per_scale * (5 + self.num_classes)), dtype=np.float32)

        batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32) # 4,150,4
        batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
        batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)

        num = 0
        if self.batch_count < self.num_batchs:
            while num < self.batch_size:
                index = self.batch_count * self.batch_size + num
                if index >= self.num_samples: index -= self.num_samples
                annotation = self.annotations[index]
                image, bboxes = self.parse_annotation(annotation)
                label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(bboxes)
                # => (52, 52, 255), (26, 26, 255), (13, 13, 255), (150, 4), (150, 4), (150, 4)

                batch_image[num, :, :, :] = image
                batch_label_sbbox[num, :, :, :] = label_sbbox
                batch_label_mbbox[num, :, :, :] = label_mbbox
                batch_label_lbbox[num, :, :, :] = label_lbbox
                batch_sbboxes[num, :, :] = sbboxes
                batch_mbboxes[num, :, :] = mbboxes
                batch_lbboxes[num, :, :] = lbboxes
                num += 1
            self.batch_count += 1
            # batch_smaller_target = batch_label_sbbox, batch_sbboxes
            # batch_medium_target  = batch_label_mbbox, batch_mbboxes
            # batch_larger_target  = batch_label_lbbox, batch_lbboxes

            # return batch_image, (batch_smaller_target, batch_medium_target, batch_larger_target)
            return batch_image, batch_label_sbbox, batch_sbboxes, batch_label_mbbox, batch_mbboxes, batch_label_lbbox, batch_lbboxes
        else:
            self.batch_count = 0
            np.random.shuffle(self.annotations)
            raise StopIteration

    def random_horizontal_flip(self, image, bboxes):

        if random.random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            bboxes[:, [0,2]] = w - bboxes[:, [2,0]]

        return image, bboxes

    def random_crop(self, image, bboxes):

        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

            image = image[crop_ymin : crop_ymax, crop_xmin : crop_xmax]

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

        return image, bboxes

    def random_translate(self, image, bboxes):

        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

        return image, bboxes

    def parse_annotation(self, annotation):

        line = annotation.split()
        image_path = line[0]
        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " %image_path)
        image = cv2.imread(image_path)
        bboxes = np.array([list(map(int, box.split(','))) for box in line[1:]])

        if self.data_aug:
            image, bboxes = self.random_horizontal_flip(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_crop(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_translate(np.copy(image), np.copy(bboxes))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, bboxes = utils.image_preporcess(np.copy(image), [self.train_input_size, self.train_input_size], np.copy(bboxes))
        '''
        print('dataset.py/parse_annotation')
        print(bboxes.shape)
        dataset.py/parse_annotation
        (9, 5)
        dataset.py/parse_annotation
        (4, 5)
        dataset.py/parse_annotation
        (8, 5)
        dataset.py/parse_annotation
        (2, 5)

        Four images per batch.
        Each contain 9, 4, 8, 2 objects respectively.
        '''
        return image, bboxes

    def bbox_iou(self, boxes1, boxes2):
        # print('core/dataset.py/bbox_iou')
        '''
        boxes1.shape: (1, 4)
        boxes2.shape: (3, 4)
        '''
        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area

        return inter_area / union_area

    def preprocess_true_boxes(self, bboxes):
        '''
        This function works per image
        Thus, called 4 times when batch_size = 4
        print('preprocess_true_boxes')
        print(bboxes.shape)
        preprocess_true_boxes
        (9, 5) => # of bboxes per image is 9, each with xywh+conf.score
        preprocess_true_boxes
        (4, 5)
        preprocess_true_boxes
        (8, 5)
        preprocess_true_boxes
        (2, 5)
        '''
        # label = [np.zeros((self.train_output_sizes[i], self.train_output_sizes[i], self.anchor_per_scale,
                        #    5 + self.num_classes)) for i in range(3)]
        label_small = np.zeros((self.train_output_sizes[0], self.train_output_sizes[0], self.anchor_per_scale, 5 + self.num_classes))
        label_medium = np.zeros((self.train_output_sizes[1], self.train_output_sizes[1], self.anchor_per_scale, 5 + self.num_classes)) 
        label_large = np.zeros((self.train_output_sizes[2], self.train_output_sizes[2], self.anchor_per_scale, 5 + self.num_classes))

        # print('label')
        # print(label[0].shape) => (52, 52, 3, 85)
        # print(label[1].shape) => (26, 26, 3, 85)    
        # print(label[2].shape) => (13, 13, 3, 85)
        # label := [(52, 52, 3, 85), (26, 26, 3, 85), (13, 13, 3, 85)]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
        bbox_count = np.zeros((3,)) # counter for each scale

        for bbox in bboxes:
            '''
            print(bbox.shape) 
                => (5,)
            '''
            bbox_coor = bbox[:4]
            bbox_class_ind = bbox[4]
            '''
            print(bbox_class_ind)
            => 80 classes for coco
            '''
            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]
            # print(self.strides) => [ 8 16 32]
            '''
            print('preprocess_true_boxes')
            print(smooth_onehot.shape) => (80,)
            print(bbox_xywh.shape)     => (4,)
            print(bbox_xywh_scaled.shape) => (3, 4)
            '''

            iou = []
            exist_positive = False
            for i in range(3): # Iterate through three scale levels (s,m,l)
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = self.anchors[i]
                '''
                print('anchors_xywh')
                print(anchors_xywh.shape) => (3,4) 
                3 anchors per each scale. Each anchor specified by xywh coordinate.
                '''
                # Finding best scale leve for the given bounding box
                # print(bbox_xywh_scaled[i][np.newaxis, :].shape) => (1, 4)
                iou_scale = self.bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                '''
                print('iou_scale') => iou_scale
                print(iou_scale.shape) => (3,)
                print(iou_scale)       => [0.12156238 0.1853328  0.46638767]
                '''
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3
                '''
                print('iou_mask')
                print(iou_mask)
                e.g.
                [False False  True]
                ...
                [ True  True False]
                [ True  True  True]
                ...
                '''
                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)
                    # label[i][yind, xind, iou_mask, :] = 0
                    # label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    # label[i][yind, xind, iou_mask, 4:5] = 1.0
                    # label[i][yind, xind, iou_mask, 5:] = smooth_onehot
                    if i == 0:
                        label_small[yind, xind, iou_mask, :] = 0
                        label_small[yind, xind, iou_mask, 0:4] = bbox_xywh
                        label_small[yind, xind, iou_mask, 4:5] = 1.0
                        label_small[yind, xind, iou_mask, 5:] = smooth_onehot
                    elif i == 1:
                        label_medium[yind, xind, iou_mask, :] = 0
                        label_medium[yind, xind, iou_mask, 0:4] = bbox_xywh
                        label_medium[yind, xind, iou_mask, 4:5] = 1.0
                        label_medium[yind, xind, iou_mask, 5:] = smooth_onehot
                    else:
                        label_large[yind, xind, iou_mask, :] = 0
                        label_large[yind, xind, iou_mask, 0:4] = bbox_xywh
                        label_large[yind, xind, iou_mask, 4:5] = 1.0
                        label_large[yind, xind, iou_mask, 5:] = smooth_onehot
                    
                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale) # choose best scale level 
                best_anchor = int(best_anchor_ind % self.anchor_per_scale) # choose best anchor for the scale
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                # label[best_detect][yind, xind, best_anchor, :] = 0
                # label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                # label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                # label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot
                assert(type(best_detect) == type(1))
                if best_detect == 0:
                    label_small[yind, xind, best_anchor, :] = 0
                    label_small[yind, xind, best_anchor, 0:4] = bbox_xywh
                    label_small[yind, xind, best_anchor, 4:5] = 1.0
                    label_small[yind, xind, best_anchor, 5:] = smooth_onehot
                elif best_detect == 1:
                    label_medium[yind, xind, best_anchor, :] = 0
                    label_medium[yind, xind, best_anchor, 0:4] = bbox_xywh
                    label_medium[yind, xind, best_anchor, 4:5] = 1.0
                    label_medium[yind, xind, best_anchor, 5:] = smooth_onehot
                else:
                    label_large[yind, xind, best_anchor, :] = 0
                    label_large[yind, xind, best_anchor, 0:4] = bbox_xywh
                    label_large[yind, xind, best_anchor, 4:5] = 1.0
                    label_large[yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
        # label_sbbox, label_mbbox, label_lbbox = label
        label_sbbox = tf.reshape(label_small, [self.train_output_sizes[0],self.train_output_sizes[0],-1])
        label_mbbox = tf.reshape(label_medium, [self.train_output_sizes[1],self.train_output_sizes[1],-1])
        label_lbbox = tf.reshape(label_large, [self.train_output_sizes[2],self.train_output_sizes[2],-1])
        # label_~bbox => (output_size, output_size, 255)
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        '''
        print('preprocess_true_boxes() return values')
        print([label_sbbox.shape, label_mbbox.shape, label_lbbox.shape])
        => [(52, 52, 3, 85), (26, 26, 3, 85), (13, 13, 3, 85)]
        print([sbboxes.shape, mbboxes.shape, lbboxes.shape])
        => [(150, 4), (150, 4), (150, 4)]

        ~bboxes val contains xywh info for all bboxes for each scale level.
        Theoretically, one can derive ~bboxes from label_~bbox val.
        '''
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def __len__(self):
        return self.num_batchs




