#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm
@file: config.py
@time: 2018/4/2 13:50
"""
import tensorflow as tf
import os

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float('learning_rate',0.01,'learning rate')
tf.app.flags.DEFINE_integer('batch_size',2,'batch size')
tf.app.flags.DEFINE_string('mode','train','mode')
tf.app.flags.DEFINE_float('decay_rate',0.99,'decay rate')
tf.app.flags.DEFINE_integer('decay_steps',100,'decay_steps')
tf.app.flags.DEFINE_string('optimizer','adam','optimizer')
tf.app.flags.DEFINE_integer('rpn_steps',1000,'rpn train_step')
tf.app.flags.DEFINE_integer('head_steps',1000,'head step')
tf.app.flags.DEFINE_integer('all_steps',10000,'all step')


class Config(object):
    def __init__(self,FLAGS):
        self.mode = FLAGS.mode
        self.batch_size = FLAGS.batch_size

        self.learning_rate = FLAGS.learning_rate
        self.decay_steps = FLAGS.decay_steps
        self.decay_rate = FLAGS.decay_rate
        self.rpn_steps = FLAGS.rpn_steps
        self.head_steps = FLAGS.head_steps
        self.all_steps = FLAGS.all_steps

        self.optimizer = FLAGS.optimizer

        # backbone strides
        self.strides = [2,4,8,16,16]


        self.pool_size = (7,7)
        self.target_size = (32,32)

        self.enter_flow_filters = (32,64,182)
        self.middle_flow_filters = 728
        assert self.enter_flow_filters[-1]==self.middle_flow_filters/4,'channel should be match'
        self.middle_flow_blocks = 16

        self.anchor_size = (4,8,16,32,64)
        self.anchor_ratio = (0.5,1,2)

        self.rpn_confidence = 0.7

        self.num_class = 1+1
        self.max_instance = 500
        self.image_size = (512,512)
        self.feature_map_size = (int(self.image_size[0]/16),int(self.image_size[1]/16))
        self.num_anchors = len(self.anchor_size)*len(self.anchor_ratio)*self.feature_map_size[0]*self.feature_map_size[1]
        self.neg_ratio = self.max_instance/(0.6*self.num_anchors)
        self.anchor_per_position = len(self.anchor_size)*len(self.anchor_ratio)


        # data_dir
        # confirm your data dir
        self.data_dir = '/data1/zhaozhonghua/data/DSB/'
        self.train_dir = self.data_dir+'stage1_train/'
        self.test_dir = self.data_dir +'stage1_test/'
        self.ckpt_dir = self.data_dir+'ckpt/'
        if not os.path.isdir(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)



        # get_shape
        # do not change anything here
        self.image_shape = (self.batch_size,)+self.image_size+(3,)
        self.size_shape = (self.batch_size,2)
        self.gt_anchor_shape = (self.batch_size,)+self.feature_map_size+(self.anchor_per_position,1)
        self.gt_label_shape = (self.batch_size,)+self.feature_map_size+(self.anchor_per_position,self.num_class)
        self.gt_box_shape = (self.batch_size,)+self.feature_map_size+(self.anchor_per_position,4)
        self.gt_mask_shape = (self.batch_size,)+self.feature_map_size+(self.anchor_per_position,)+self.target_size


CONFIG = Config(FLAGS)








if __name__ == '__main__':
    pass
