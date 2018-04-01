#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm
@file: model.py.py
@time: 2018/3/30 16:56
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from . import modellib
from . import utils

FLAGS = tf.app.flags.FLAGS

class Model(object):
    def __init__(self,config):
        enter_flow = modellib.EnterFlow(filter_list=[32,64,128])
        middle_flow = modellib.MiddleFlow(filters=728,num_blocks=4)
        backbone = modellib.BackBone(enter_flow=enter_flow,middle_flow=middle_flow)
        anchor_per_position = len(config.anchor_size)*len(config.anchor_ratio)
        rpn_layer = modellib.RPN(anchor_per_position)
        roi_layer = modellib.PyramidROILayer(config.pool_size)
        boxes_layer = modellib.MaskBboxBranch(config.pool_size,num_class=config.num_class)
        mask_layer = modellib.MaskLayer(strides=[2,4,8,16,16])
        self.mask_rcnn = modellib.MaskRcnn(backbone,rpn_layer,roi_layer,boxes_layer,mask_layer)
        self.config = config

    def build_graph(self,iterator):
        box_logits, box_probs, box_positions, mask_class_logits, bbox, mask = self.mask_rcnn.inference(iterator['images'])
        self.rpn_variables = self.mask_rcnn.rpn_variables
        self.head_variables = self.mask_rcnn.head_variables
        self.all_variables = self.mask_rcnn.trainable_variables
        if self.config.mode == 'train':
            rpn_loss = modellib.compute_rpn_loss(box_logits,box_positions,iterator['gt_labels'],iterator['gt_box'])
            mask_loss = modellib.compute_mask_rcnn_loss(mask_class_logits,bbox,mask,iterator['mask'])

    def train_rpn(self):
        pass

    def train_head(self):
        pass

    def train_all(self):
        pass



if __name__ == '__main__':
    pass