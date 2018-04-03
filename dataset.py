#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm
@file: dataset.py.py
@time: 2018/4/2 14:21
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import utils
import config
import os
from PIL import Image
import numpy as np
import scipy.misc

FLAGS = config.FLAGS

CONFIG = config.CONFIG
TRAIN_IMAGE_IDS = os.listdir(CONFIG.train_dir)
TEST_IMAGE_IDS = os.listdir(CONFIG.test_dir)


class tf_func(object):
    def __init__(self,Tout):
        self.Tout = Tout

    def __call__(self,func):
        return true_dec(func,self.Tout)

class true_dec(object):
    def __init__(self,func,Tout):
        self.func = func
        self.Tout = Tout
    def __call__(self,inputs):
        result = tf.py_func(self.func, inp=[inputs], Tout=self.Tout)
        return result




@tf_func([np.float32,np.int32,np.float32,np.float32,np.float32,np.float32])
def process_train(image_id):
    image_file = CONFIG.train_dir + '%s/images/%s.png' % ((image_id.decode()), (image_id.decode()))
    image = Image.open(image_file)
    image = image.convert('RGB')

    # print(np.shape(image))
    image = image.resize(CONFIG.image_size)
    image_shape = np.array(np.shape(image)[:2],dtype=np.int32)


    image = np.array(image).astype(np.float32)
    masks = np.zeros((CONFIG.max_instance,CONFIG.image_size[0],CONFIG.image_size[1]),dtype=np.float32)
    # print(masks.shape)
    masks_ids = os.listdir(CONFIG.train_dir+'%s/masks/'%(image_id.decode()))
    for i,idx in enumerate(masks_ids):
        mask = Image.open(CONFIG.train_dir+'%s/masks/%s'%(image_id.decode(),idx))
        # print(np.array(mask).shape)
        mask = mask.convert('L')
        mask = mask.resize(CONFIG.image_size)
        # masks[i] = (np.array(mask)>0).astype(np.int32)
        mask = np.array(mask)
        if mask.any():
            masks[i,:] = mask
    masks = np.transpose(masks,[1,2,0])
    boxes = utils.extract_bboxes(masks)
    gt_anchors,gt_boxes,gt_labels,gt_masks = utils.boxes2anchor(image_shape = CONFIG.image_size,
                                            strides = CONFIG.strides[-1],
                                            anchor_size = CONFIG.anchor_size,
                                            anchor_ratio = CONFIG.anchor_ratio,
                                            boxes = boxes,
                                            masks = masks,
                                            target_size = CONFIG.target_size)
    # masks = scipy.misc.imresize(masks,CONFIG.image_size).astype(np.int32)
    return image,image_shape,gt_anchors,gt_labels,gt_boxes,gt_masks


def find_shape(image,image_shape,gt_anchors,gt_labels,gt_boxes,gt_masks):
    image.set_shape(CONFIG.image_shape)
    image_shape.set_shape(CONFIG.size_shape)
    gt_anchors.set_shape(CONFIG.gt_anchor_shape)
    gt_labels.set_shape(CONFIG.gt_label_shape)
    gt_boxes.set_shape(CONFIG.gt_box_shape)
    gt_masks.set_shape(CONFIG.gt_mask_shape)\

    # image = tf.reshape(image,CONFIG.image_shape)
    # image_shape =tf.reshape(image_shape,[2])
    # gt_anchors = tf.reshape(gt_anchors,CONFIG.gt_anchor_shape)
    # gt_labels = tf.reshape(gt_labels,CONFIG.gt_label_shape)
    # gt_boxes = tf.reshape(gt_boxes,CONFIG.gt_box_shape)
    # gt_masks = tf.reshape(gt_masks,CONFIG.gt_mask_shape)
    return image,image_shape,gt_anchors,gt_labels,gt_boxes,gt_masks



@tf_func([np.int32,np.int32])
def process_test(image_id):
    image_file = CONFIG.test_dir + '%s/images/%s.png' % ((image_id.decode()), (image_id.decode()))
    image = Image.open(image_file)
    image_shape = np.array(np.shape(image)[:2],dtype=np.int32)
    image = image.resize((CONFIG.image_size[1], CONFIG.image_size[0]))
    image = np.array(image).astype(np.int32)
    return image.astype(np.int32),image_shape




def build_train_dataset(image_ids):

    dataset = tf.data.Dataset.from_tensor_slices(image_ids)
    dataset = dataset.shuffle(buffer_size=1000).repeat()
    dataset = dataset.map(process_train,num_parallel_calls=16)
    # dataset = dataset.map(find_shape).batch(2)
    dataset = dataset.batch(CONFIG.batch_size)
    return dataset

if __name__ == '__main__':
    # for i in range(100):
    #     a = process_train(TRAIN_IMAGE_IDS[i].encode())
    #     for j in a:
    #         print(j.shape)

    dataset = build_train_dataset(TRAIN_IMAGE_IDS)
    iter = dataset.make_one_shot_iterator()
    a = iter.get_next()
    print(a)
    sess = tf.Session()
    for i in range(100):
        print(i)
        sess.run(a)
