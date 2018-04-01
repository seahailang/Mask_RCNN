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
    def __init__(self,config,iterator):
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
        if self.config.mode == 'train':
            self.images,self.image_sizes,self.masks,self.gt_labels,self.gt_boxes = iterator.get_next()
        else:
            self.images,self.image_sizes = iterator.get_next()


    def build_graph(self):
        box_logits, box_probs, box_positions, mask_class_logits, bbox, mask = self.mask_rcnn.inference(self.images)
        self.rpn_variables = self.mask_rcnn.rpn_variables
        self.head_variables = self.mask_rcnn.head_variables
        self.all_variables = self.mask_rcnn.trainable_variables
        self.image_net_variables = self.mask_rcnn.image_net_variables
        if self.config.mode == 'train':
            self.rpn_loss = modellib.compute_rpn_loss(box_logits,box_positions,self.gt_boxes,self.gt_labels)
            self.mask_loss = modellib.compute_mask_rcnn_loss(mask_class_logits,bbox,mask,self.masks)
            return self.rpn_loss,self.mask_loss
        else:
            return mask_class_logits,bbox,mask

    def load_weights(self,sess,ckpt,flags='all'):
        assert flags in ['all','rpn','head'],'''flags in 'all','rpn','head' '''
        if flags == 'all':
            saver = tf.train.Saver(self.all_variables)
        elif flags == 'rpn':
            saver = tf.train.Saver(self.rpn_variables)
        elif flags == 'head':
            saver = tf.train.Saver(self.head_variables)
        else:
            saver = tf.train.Saver(self.image_net_variables)
        if ckpt:
            saver.restore(sess,ckpt)
        return saver

    def optimizer(self):
        global_step = tf.train.get_or_create_global_step()
        lr = tf.train.exponential_decay(learning_rate=self.config.learning_rate,
                                        global_step=global_step,
                                        decay_rate=self.config.decay_rate,
                                        decay_steps=self.config.decay_step)
        if self.config.opt == 'adam':
            opt = tf.train.AdamOptimizer(learning_rate=lr)
        elif self.config.opt == 'adgrad':
            opt = tf.train.AdagradOptimizer(learning_rate=lr)
        else:
            opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
        return opt

    def train_rpn(self,opt,losses):
        vars = self.rpn_variables
        global_step = tf.train.get_or_create_global_step()

        grads_and_vars = opt.compute_gradients(losses,vars)
        op = opt.apply_gradients(grads_and_vars,global_step=global_step)
        return op

    def train_head(self,opt,losses):
        vars = self.head_variables
        global_step = tf.train.get_or_create_global_step()

        grads_and_vars = opt.compute_gradients(losses, vars)
        op = opt.apply_gradients(grads_and_vars, global_step=global_step)
        return op


    def train_all(self,opt,losses):
        vars = self.all_variables
        global_step = tf.train.get_or_create_global_step()

        grads_and_vars = opt.compute_gradients(losses, vars)
        op = opt.apply_gradients(grads_and_vars, global_step=global_step)
        return op


def train(config,train_dataset,val_dataset):

    assert config.mode == 'train','training flag error'
    train_it = train_dataset.make_one_shot_iterator()
    val_it = val_dataset.make_one_shot_iterator()
    iter_placeholder = tf.placeholder(tf.string,shape=[],name='init_iterator')

    iterator = tf.data.Iterator.from_string_handle(iter_placeholder,
                                                         train_it.output_types,
                                                         train_it.output_shapes)
    # val_iteration = tf.data.Iterator.from_string_handle(iter_placeholder,
    #                                                     val_it.output_types,
    #                                                     val_it.output_shapes)

    model = Model(config, iterator)
    # build_graph
    rpn_loss,mask_loss = model.build_graph()
    loss_rpn = rpn_loss['rpn_box_loss']+rpn_loss['rpn_class_loss']
    loss_mask = mask_loss['mask_box_loss']+mask_loss['mask_loss']+mask_loss['mask_class_loss']
    opt = model.optimizer()
    rpn_op = model.train_rpn(opt,loss_rpn)
    head_op = model.train_head(opt,loss_mask)
    all_op = model.train_all(opt,loss_mask)
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(logdir=config.ckpt_dir)
    init = tf.global_variables_initializer()

    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        train_handel = sess.run(train_it.string_handel())
        val_handel = sess.run(val_it.string_handle())
        ckpt = tf.train.latest_checkpoint(checkpoint_dir=config.ckpt_dir,latest_filename='mask_rcnn')
        if not ckpt:
            ckpt = tf.train.latest_checkpoint(checkpoint_dir=config.ckpt_dir,
                                              latest_filename='rpn')
        if ckpt:
            print('load weights from %s'%(ckpt))
            model.load_weights(sess,ckpt,flags='rpn')
        else:
            sess.run(init)
        global_step = tf.train.get_or_create_global_step()
        writer.add_graph(tf.get_default_graph())
        for step in range(config.rnp_steps):

            sess.run([rpn_loss,loss_rpn,rpn_op],
                     feed_dict={iter_placeholder:train_handel})
            if step%10 == 0:
                rpn_l,l_rpn,summary_str=sess.run([rpn_loss,loss_rpn,summary_op],
                                                       feed_dict={iter_placeholder:val_handel})

                writer.add_summary(summary_str)

                saver.save(sess,
                           config.ckpt_dir,
                           global_step=global_step,
                           latest_filename='rpn')

        for step in range(config.head_steps):
            sess.run([mask_loss, loss_mask, head_op],
                     feed_dict={iter_placeholder: train_handel})
            if step % 10 == 0:
                mask_l, l_mask, summary_str = sess.run([mask_loss, loss_mask, summary_op],
                                                           feed_dict={iter_placeholder: val_handel})

                writer.add_summary(summary_str)

                saver.save(sess,
                           config.ckpt_dir,
                           global_step=global_step,
                           latest_filename='mask_rcnn')

        for step in range(config.all_steps):
            sess.run([mask_loss, loss_mask, all_op],
                     feed_dict={iter_placeholder: train_handel})
            if step % 10 == 0:
                mask_l, l_mask, summary_str = sess.run([mask_loss, loss_mask, summary_op],
                                                           feed_dict={iter_placeholder: val_handel})

                writer.add_summary(summary_str)

                saver.save(sess,
                           config.ckpt_dir,
                           global_step=global_step,
                           latest_filename='mask_rcnn')










if __name__ == '__main__':
    pass