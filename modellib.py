#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm
@file: modellib.py
@time: 2018/3/29 14:48
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import math
import utils
FLAGS = tf.app.flags.FLAGS



#Todo Xception architecture
class StrideBlock(object):
    def __init__(self,name='stage1',filters=16):
        self.name = name
        self.layers = []
        self.layers.append(tf.layers.SeparableConv2D(filters=filters,
                                                     kernel_size=(3, 3),
                                                     padding='same',
                                                     activation=tf.nn.relu))
        self.layers.append(tf.layers.SeparableConv2D(filters=filters,
                                                     kernel_size=(3, 3),
                                                     dilation_rate=(2, 2),
                                                     padding='same',
                                                     activation=tf.nn.relu))
        self.layers.append(tf.layers.SeparableConv2D(filters=filters,
                                                     kernel_size=(3, 3),
                                                     dilation_rate=(3, 3),
                                                     padding='same',
                                                     activation=tf.nn.relu))
        self.layers.append(tf.layers.SeparableConv2D(filters=filters,
                                                     kernel_size=(3, 3),
                                                     dilation_rate=(5, 5),
                                                     padding='same',
                                                     activation=tf.nn.relu))
        self.layers.append(tf.layers.SeparableConv2D(filters=filters*4,
                                                     kernel_size=(3,3),
                                                     strides=(2,2),
                                                     padding='same'))

    def __call__(self, X):
        with tf.variable_scope(self.name):
            for i in range(len(self.layers)):
                X = tf.nn.batch_normalization(self.layers[i](X))
        return X
    @property
    def trainable_variables(self):
        variables=[]
        for i in range(len(self.layers)):
            variables.extend(self.layers[i].trainable_variables)
        return variables

        # with tf.variable_scope(name):



class EnterFlow(object):
    def __init__(self,filter_list=[32,64,128]):
        self.in_layer1 = tf.layers.Conv2D(filters=32,
                                          kernel_size=(3,3),
                                          strides=(2,2),
                                          padding='same',
                                          activation=tf.nn.relu)
        self.in_layer2 = tf.layers.Conv2D(filters=64,
                                          kernel_size=(3,3),
                                          padding='same',
                                          activation=tf.nn.relu)
        self.blocks = []
        self.bridges = []
        for i,filters in filter_list:
            self.blocks.append(StrideBlock(name='stage%d'%i,
                                           filters=filters))
            self.bridges.append(tf.layers.Conv2D(filters=filters*4,
                                                 kernel_size=(1,1),
                                                 strides=(2,2),
                                                 padding='same',
                                                 activation=tf.nn.relu))
    def __call__(self,X):
        block_output = []
        # stride 2
        X = self.in_layer1(X)
        X = self.in_layer2(X)
        block_output.append(X)
        B = X
        # stride 16
        for i in range(len(self.blocks)):
            X = self.blocks[i](X)
            B = self.bridges[i](B)
            X = tf.add(X,B)
            block_output.append(X)
        # block_output with stride 2,4,8,16
        return block_output,X
    @property
    def trainable_variables(self):
        variables = []
        variables.extend(self.in_layer1.trainable_variables)
        variables.extend(self.in_layer2.trainable_variables)
        for i in range(len(self.blocks)):
            variables.extend(self.blocks[i].trainable_variables)
            variables.extend(self.bridges[i].trainable_variables)
        return variables


class ResBlock(object):
    def __init__(self,name='block',filters=728):
        self.name = name
        self.layers = []
        self.layers.append(tf.layers.SeparableConv2D(filters=filters,
                                                     kernel_size=(3,3),
                                                     padding='same',
                                                     activation=tf.nn.relu))
        self.layers.append(tf.layers.SeparableConv2D(filters=filters,
                                                     kernel_size=(3,3),
                                                     padding='same',
                                                     activation=tf.nn.relu))
        self.layers.append(tf.layers.SeparableConv2D(filters=filters,
                                                     kernel_size=(3,3),
                                                     padding='same',
                                                     activation=tf.nn.relu))
    def __call__(self,X):
        C = X
        with tf.variable_scope(self.name):
            for i in range(3):
                C = self.layers[i](C)
        return C+X

    @property
    def trainable_variables(self):
        variables = []
        for i in range(3):
            variables.extend(self.layers[i].trainable_variables)
        return variables

class MiddleFlow(object):
    def __init__(self,filters,num_blocks):
        self.blocks = []
        self.num_blocks = num_blocks
        for i in range(num_blocks):
            self.blocks.append(ResBlock(name='block%d'%(i),filters=filters))
    def __call__(self,X):
        for i in range(self.num_blocks):
            X = self.blocks[i](X)
        return X
    @property
    def trainable_variables(self):
        variables = []
        for i in range(self.num_blocks):
            variables.extend(self.blocks[i].trainable_variables)
        return variables


class BackBone(object):
    def __init__(self,enter_flow=EnterFlow(),middle_flow=MiddleFlow(728,16)):
        self.enter_flow = enter_flow
        self.middle_flow = middle_flow
    def __call__(self,X):
        C,X = self.enter_flow(X)
        X = self.middle_flow(X)
        C.append(X)
        return C,X
    @property
    def trainable_variables(self):
        return self.enter_flow.trainable_variables\
               +self.middle_flow.trainable_variables


class RPN(object):
    def __init__(self,anchor_per_position):
        self.num_achors = anchor_per_position
        # self.layers = []
        self.layer = tf.layers.Conv2D(filters=512,
                                            kernel_size=(3,3),
                                            padding='same',
                                            activation=tf.nn.relu)
        self.rpn_class_layers = tf.layers.Conv2D(anchor_per_position,
                                            kernel_size=(1,1),
                                            activation=None)
        self.rpn_bbox_layers = tf.layers.Conv2D(4*anchor_per_position,
                                                kernel_size = (1,1),
                                                activation=None)
    def __call__(self,X):
        shared = self.layer(X)
        rpn_logits = self.rpn_class_layers(shared)
        rpn_logits= tf.reshape(rpn_logits,shape=[rpn_logits.shape[0],-1,1])
        rpn_prob = tf.nn.sigmoid(rpn_logits)
        rpn_bbox_logits = self.rpn_bbox_layers(shared)
        return rpn_logits,rpn_prob,rpn_bbox_logits

    @property
    def trainable_variables(self):
        return self.layer.trainable_variables \
               +self.rpn_class_layers.trainable_variables\
               +self.rpn_bbox_layers.trainable_variables

def crop_and_resize_with_batch(image_batch,boxes_batch,crop_size):
    batch_size = image_batch.shape[0]
    idx = tf.cast(tf.range(batch_size),tf.int32)
    idx = tf.expand_dims(idx,axis=1)
    idx = tf.tile(idx,[1,boxes_batch.shape[1]])
    idx = tf.reshape(idx,[-1])
    boxes_batch = tf.reshape(boxes_batch,[-1,boxes_batch.shape[-1]])
    return tf.image.crop_and_resize(image_batch,boxes_batch,box_ind=idx,crop_size=crop_size)

class PyramidROILayer(object):
    def __init__(self,pool_size=(7,7)):
        self.pool_size=pool_size
        # self.image_size = image_size
    def __call__(self,features,bboxes):
        # 目前只处理一张图片
        bboxes = tf.stop_gradient(bboxes)
        # assert len(features) == len(self.strides), 'strides should match the feature map'
        # assert features[0].shape[0] == 1 ,'can only process 1 image once'
        rois = []

        for i,strides in enumerate(len(features)):
            # each roi with shapes num_bboxes,pool_size[0],pool_size[1],raw_channel
            rois.append(crop_and_resize_with_batch(features[i],
                                                   bboxes,
                                                   crop_size=self.pool_size))
        # ROI with shapes of num_bboxes,pool_size[0],pool_size[1],all_channel
        return tf.concat(rois,axis=3)



class MaskBboxBranch(object):
    def __init__(self,pool_size,num_class):
        self.layers = []
        self.layers.append(tf.layers.Conv2D(filters=1024,
                                            kernel_size=pool_size,
                                            strides=(1,1),
                                            activation=tf.nn.relu))
        self.layers.append(tf.layers.Conv2D(filters=1024,
                                            kernel_size=(1,1),
                                            strides=(1,1),
                                            activation=tf.nn.relu))
        self.bbox_layer = tf.layers.Conv2D(filters=4,
                                            kernel_size=(1,1))
        self.mask_class_layer = tf.layers.Conv2D(filters=num_class,
                                                 kernel_size=(1,1),
                                                 activation=None)
        self.roi_layer = PyramidROILayer(pool_size)
    def __call__(self,rois):
        x = self.roi_layer(rois)
        for i in range(2):
            x = self.layers[i](x)
        bbox = self.bbox_layer(x)
        mask_class_logits = self.mask_class_layer(x)
        mask_prob = tf.nn.softmax(mask_class_logits,axis=-1)
        return mask_class_logits,mask_prob,bbox

    @property
    def trainable_variables(self):
        variables = []
        for i in range(2):
            variables.append(self.layers[i].trainable_variables)
        variables.append((self.bbox_layer.trainable_variables))
        variables.append(self.mask_class_layer.trainable_variables)
        return variables

class MaskLayer(object):
    def __init__(self,strides=[2,4,8,16,16]):
        self.strides = []
        for s in strides:
            self.strides.append(s/2)
        self.layers = []
        self.layers.append(tf.layers.Conv2D(filters=256,
                                            kernel_size=(3,3),
                                            strides=(1,1),
                                            padding='same',
                                            activation=tf.nn.relu))
        self.layers.append(tf.layers.Conv2D(filters=256,
                                            kernel_size=(3, 3),
                                            strides=(1, 1),
                                            padding='same',
                                            activation=tf.nn.relu))
        self.layers.append(tf.layers.Conv2DTranspose(filters=256,
                                                     kernel_size=(3,3),
                                                     strides=(2,2),
                                                     padding='same',
                                                     activation=tf.nn.relu))
        self.layers.append(tf.layers.Conv2D(filters=80,
                                            kernel_size=(3, 3),
                                            strides=(1, 1),
                                            padding='same',
                                            activation=tf.nn.relu))
        self.layers.append(tf.layers.Conv2D(filters=1,
                                            kernel_size=(3, 3),
                                            strides=(1, 1),
                                            padding='same',
                                            activation=tf.nn.relu))
    def __call__(self,feature_maps,bbox):
        assert len(feature_maps)==len(self.strides)
        # l = len(self.strides)
        for i in range(len(self.strides)):
            feature_maps[i] = tf.image.resize_bilinear(feature_maps[i],size=feature_maps[i].shape*self.strides)
        x = tf.concat(feature_maps,axis=3)
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        x = tf.nn.sigmoid(x)
        return x

    @property
    def trainable_variables(self):
        variables=[]
        for i in range(len(self.layers)):
            variables.append(self.layers[i].trainable_variables)
        return variables


# def anchor2boxes(image_shape,strides,anchor_size,anchor_ratio):





def compute_rpn_loss(rpn_logits,rpn_boxes,gt_labels,gt_boxes):
    l1 = tf.losses.log_loss(labels=gt_labels,predictions=rpn_logits)

    # 非boxes部分不计算损失,所以要multiply with labels
    rpn_boxes_with_gt = tf.multiply(rpn_boxes,gt_labels)
    l2 = tf.losses.mean_squared_error(labels=gt_boxes,predictions = rpn_boxes_with_gt)
    return {'rpn_class_loss':l1,'rpn_box_loss':l2}


# # Todo compute mask loss with boxes,only mask in a box should be computed
# # something unsolved: can we compute loss with a specific instance rather than global mask
#
def crop_mask_and_resize(maskes_batches,boxes,crop_size):
    batch_size = maskes_batches.shape[0]
    num_instance = maskes_batches.shape[1]
    channels = maskes_batches.shape[-1]
    num_boxes = boxes.shape[1]

    # boxes with [b num_instance num_boxes 4]
    boxes = tf.stop_gradient(boxes)
    boxes = tf.tile(tf.expand_dims(boxes,axis=1),[1,maskes_batches.shape[1],1,1])
    # boxes with [b*num_instance ,num_boxes,4]
    boxes = tf.reshape(boxes,[-1,boxes.shape[-2],boxes.shape[-1]])
    # maskes with [b*num_instance, w,h,channel]
    maskes = tf.reshape(maskes_batches,[-1]+list(maskes_batches.shape[-3:]))
    # crops with [b*num_instance*numboxes,w,h,channel]
    crops = crop_and_resize_with_batch(maskes,boxes,crop_size)
    #crop_mask [b,num_instance,numboxes,w,h,channel]
    crop_mask = tf.reshape(crops,[batch_size,num_instance,num_boxes,crop_size[0],crop_size[1],channels])
    # crop_mask_sum [b,num_instance,numboxes]
    crop_mask_sum = tf.reduce_sum(crop_mask,axis=[3,4,5])

    # 保留每个predict_boxes对应的最大的那个mask
    # 这里可能会出问题,难以保证最大的只有一个
    mask_mask = tf.equal(crop_mask_sum, tf.reduce_max(crop_mask_sum, axis=1, keepdims=True))
    mask_gt = tf.boolean_mask(crop_mask,mask_mask)
    return mask_gt,mask_mask


# 这里计算mask_loss
def compute_mask_loss(pd_masks,boxes,gt_maskes):
    # gt_boxed_mask [batch_size*num_boxes,crop_size[0],crop_size[1],1]
    gt_boxed_masks, mask_mask = crop_mask_and_resize(gt_maskes, boxes, crop_size=(32, 32))
    # boxed_mask [batch_size*num_boxes,crop_size[0],crop_size[1],1],
    boxed_masks = crop_and_resize_with_batch(pd_masks,boxes,crop_size=(32,32))

    # 上面的boxed_mask其实已经是predict的目标了,但是因为boolean_mask这一步操作可能会导致维度步匹配
    # 所以这里又加了一部分
    boxed_masks = tf.reshape(boxed_masks,[pd_masks.shape[0],1,boxes.shape[1],32,32,1])
    boxed_masks = tf.tile(boxed_masks,[1,pd_masks.shape[1],1,1,1,1])
    mask_mask = tf.reshape(mask_mask,[-1])
    boxed_masks = tf.boolean_mask(boxed_masks,mask_mask)
    l = tf.losses.log_loss(labels=gt_boxed_masks,predictions=boxed_masks)
    return l


def compute_bbox_class_loss(boxes,gt_masks):
    return 0


def compute_bbox_regression_loss(boxes,gt_masks):
    return 0

def compute_mask_rcnn_loss(boxes_class_logits,boxes,masks,gt_masks):
    l1 = compute_bbox_class_loss(boxes_class_logits,gt_masks)
    l2 = compute_bbox_regression_loss(boxes,gt_masks)
    l3 = compute_mask_loss(masks,boxes,gt_masks)
    return {'mask_class_loss':l1,'mask_box_loss':l2,'mask_loss':l3}



class MaskRcnn(object):
    def __init__(self,backbone,rpn_layer,roi_layer,boxes_layer,mask_layer):
        self.backbone = backbone
        self.roi_layer = roi_layer
        self.rpn_layer = rpn_layer
        self.boxes_layer = boxes_layer
        self.mask_layer = mask_layer
    def inference(self,images):
        feature_maps,X = self.backbone(images)
        box_logits,box_probs,box_positions = self.rpn_layer(X)
        rois = self.roi_layer(feature_maps,box_positions)
        mask_class_logits, mask_prob, bbox = self.boxes_layer(rois)
        mask = self.mask_layer(feature_maps,bbox)
        return box_logits,box_probs,box_positions,mask_class_logits,bbox,mask

    @property
    def rpn_variables(self):
        return self.backbone.trainable_variables+self.rpn_layer.trainable_variables

    @property
    def head_variables(self):
        return self.boxes_layer.trainable_variables+self.mask_layer.trainable_variables
    @property
    def image_net_variables(self):
        return self.backbone.trainable_variables

    @property
    def trainable_variables(self):
        return self.rpn_variables+self.head_variables















if __name__ == '__main__':
    pass