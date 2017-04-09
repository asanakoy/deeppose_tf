# Copyright (c) 2016 Artsiom Sanakoyeu

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def training_convnet(net, loss_op, fc_lr, conv_lr, optimizer_type='adagrad',
                     trace_gradients=False):
    with net.graph.as_default():
        print('Creating optimizer {}'.format(optimizer_type))
        if optimizer_type == 'adagrad':
            conv_optimizer = tf.train.AdagradOptimizer(conv_lr,
                                                       initial_accumulator_value=0.0001)
            fc_optimizer = tf.train.AdagradOptimizer(fc_lr,
                                                     initial_accumulator_value=0.0001)
        elif optimizer_type == 'sgd':
            conv_optimizer = tf.train.GradientDescentOptimizer(conv_lr)
            fc_optimizer = tf.train.GradientDescentOptimizer(fc_lr)
        elif optimizer_type == 'momentum':
            conv_optimizer = tf.train.MomentumOptimizer(conv_lr, momentum=0.9)
            fc_optimizer = tf.train.MomentumOptimizer(fc_lr, momentum=0.9)
        else:
            raise ValueError('Unknown optimizer type {}'.format(optimizer_type))

        print('Conv LR: {}, FC LR: {}'.format(conv_lr, fc_lr))

        conv_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'conv')
        fc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'fc')

        assert len(conv_vars) + len(fc_vars) == \
            len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)),\
            'You dont train all the variables'

        grads = tf.gradients(loss_op, conv_vars + fc_vars)
        conv_grads = grads[:len(conv_vars)]
        fc_grads = grads[len(conv_vars):]
        assert len(conv_grads) == len(conv_vars)
        assert len(fc_grads) == len(fc_vars)

        with tf.name_scope('grad_norms'):
            for v, grad in zip(conv_vars + fc_vars, grads):
                if grad is not None:
                    grad_norm_op = tf.nn.l2_loss(grad, name=format(v.name[:-2]))
                    tf.add_to_collection('grads', grad_norm_op)
                    if trace_gradients:
                        tf.summary.scalar(grad_norm_op.name, grad_norm_op)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            conv_tran_op = conv_optimizer.apply_gradients(zip(conv_grads, conv_vars), name='conv_train_op')
        fc_tran_op = fc_optimizer.apply_gradients(zip(fc_grads, fc_vars),
                                                  global_step=net.global_iter_counter, name='fc_train_op')
        return tf.group(conv_tran_op, fc_tran_op)
