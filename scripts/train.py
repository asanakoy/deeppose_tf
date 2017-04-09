#!/usr/bin/env python
# Copyright (c) 2016 Artsiom Sanakoyeu

from __future__ import division
from chainer import iterators
import cmd_options
import dataset
import os
import time
import regressionnet
import tensorflow as tf
import copy
from tqdm import tqdm
import numpy as np
import math
import pprint
import datetime

from regressionnet import evaluate_pcp, create_sumamry


def evaluate(net, pose_loss_op, test_iterator, summary_writer, tag='test/pose_loss'):
    test_it = copy.copy(test_iterator)
    total_loss = 0.0
    cnt = 0
    num_batches = int(math.ceil(len(test_it.dataset) / test_it.batch_size))
    print len(test_it.dataset)
    for batch in tqdm(test_it, total=num_batches):
        feed_dict = regressionnet.fill_joint_feed_dict(net,
                                                       regressionnet.batch2feeds(batch)[:3],
                                                       conv_lr=0.0,
                                                       fc_lr=0.0,
                                                       phase='test')
        global_step, loss_value = net.sess.run([net.global_iter_counter, pose_loss_op],
                                               feed_dict=feed_dict)
        total_loss += loss_value * len(batch)
        cnt += len(batch)
    avg_loss = total_loss / len(test_it.dataset)
    print 'Step {} {} = {:.3f}'.format(global_step, tag, avg_loss)
    summary_writer.add_summary(create_sumamry(tag, avg_loss),
                               global_step=global_step)
    assert cnt == 1000, 'cnt = {}'.format(cnt)


def train_loop(net, saver, loss_op, pose_loss_op, train_op, dataset_name, train_iterator, test_iterator,
               val_iterator=None,
               max_iter=None,
               test_step=None,
               snapshot_step=None,
               log_step=1,
               batch_size=None,
               conv_lr=None,
               fc_lr=None,
               fix_conv_iter=None,
               output_dir='results',
               ):

    summary_step = 50

    with net.graph.as_default():
        summary_writer = tf.summary.FileWriter(output_dir, net.sess.graph)
        summary_op = tf.summary.merge_all()
        fc_train_op = net.graph.get_operation_by_name('fc_train_op')
    global_step = None

    for step in xrange(max_iter + 1):

        # test, snapshot
        if step % test_step == 0 or step + 1 == max_iter or step == fix_conv_iter:
            global_step = net.sess.run(net.global_iter_counter)
            evaluate_pcp(net, pose_loss_op, test_iterator, summary_writer,
                         dataset_name=dataset_name,
                         tag_prefix='test')
            if val_iterator is not None:
                evaluate_pcp(net, pose_loss_op, val_iterator, summary_writer,
                             dataset_name=dataset_name,
                             tag_prefix='val')

        if step % snapshot_step == 0 and step > 1:
            checkpoint_prefix = os.path.join(output_dir, 'checkpoint')
            assert global_step is not None
            saver.save(net.sess, checkpoint_prefix, global_step=global_step)
        if step == max_iter:
            break

        # training
        start_time = time.time()
        feed_dict = regressionnet.fill_joint_feed_dict(net,
                                                       regressionnet.batch2feeds(train_iterator.next())[:3],
                                                       conv_lr=conv_lr,
                                                       fc_lr=fc_lr,
                                                       phase='train')
        if step < fix_conv_iter:
            feed_dict['lr/conv_lr:0'] = 0.0

        if step < fix_conv_iter:
            cur_train_op = fc_train_op
        else:
            cur_train_op = train_op

        if step % summary_step == 0:
            global_step, summary_str, _, loss_value = net.sess.run(
                [net.global_iter_counter,
                 summary_op,
                 cur_train_op,
                 pose_loss_op],
                feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=global_step)
        else:
            global_step, _, loss_value = net.sess.run(
                [net.global_iter_counter, cur_train_op, pose_loss_op],
                feed_dict=feed_dict)

        duration = time.time() - start_time

        if step % log_step == 0 or step + 1 == max_iter:
            print('Step %d: train/pose_loss = %.2f (%.3f s, %.2f im/s)'
                  % (global_step, loss_value, duration,
                     batch_size // duration))


def main(argv):
    """
    Run training of the Deeppose stg-1
    """
    args = cmd_options.get_arguments(argv)
    if not os.path.exists(args.o_dir):
        os.makedirs(args.o_dir)
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    with open(os.path.join(args.o_dir, 'params.dump_{}.txt'.format(suffix)), 'w') as f:
        f.write('{}\n'.format(pprint.pformat(args)))

    net, loss_op, pose_loss_op, train_op = regressionnet.create_regression_net(
        n_joints=args.n_joints,
        init_snapshot_path=args.snapshot,
        is_resume=args.resume,
        reset_iter_counter=args.reset_iter_counter,
        reset_moving_averages=args.reset_moving_averages,
        optimizer_type=args.optimizer,
        gpu_memory_fraction=0.32,  # Set how much GPU memory to reserve for the network
        net_type=args.net_type)
    with net.graph.as_default():
        saver = tf.train.Saver()

    print 'args.resume: {}\nargs.snapshot: {}'.format(args.resume, args.snapshot)
    bbox_extension_range = (args.bbox_extension_min, args.bbox_extension_max)
    if bbox_extension_range[0] is None or bbox_extension_range[1] is None:
        bbox_extension_range = None
        test_bbox_extension_range = None
    else:
        test_bbox_extension_range = (bbox_extension_range[1], bbox_extension_range[1])

    train_dataset = dataset.PoseDataset(
        args.train_csv_fn, args.img_path_prefix, args.im_size,
        fliplr=args.fliplr,
        rotate=args.rotate,
        rotate_range=args.rotate_range,
        shift=args.shift,
        bbox_extension_range=bbox_extension_range,
        min_dim=args.min_dim,
        coord_normalize=args.coord_normalize,
        gcn=args.gcn,
        fname_index=args.fname_index,
        joint_index=args.joint_index,
        symmetric_joints=args.symmetric_joints,
        ignore_label=args.ignore_label,
        should_downscale_images=args.should_downscale_images,
        downscale_height=args.downscale_height
    )
    test_dataset = dataset.PoseDataset(
        args.test_csv_fn, args.img_path_prefix, args.im_size,
        fliplr=False, rotate=False,
        shift=None,
        bbox_extension_range=test_bbox_extension_range,
        coord_normalize=args.coord_normalize,
        gcn=args.gcn,
        fname_index=args.fname_index,
        joint_index=args.joint_index,
        symmetric_joints=args.symmetric_joints,
        ignore_label=args.ignore_label,
        should_return_bbox=True,
        should_downscale_images=args.should_downscale_images,
        downscale_height=args.downscale_height
    )

    np.random.seed(args.seed)
    train_iterator = iterators.MultiprocessIterator(train_dataset, args.batch_size,
                                                    n_processes=args.workers, n_prefetch=3)
    test_iterator = iterators.MultiprocessIterator(
        test_dataset, args.batch_size,
        repeat=False, shuffle=False,
        n_processes=1, n_prefetch=1)

    val_iterator = None
    if args.val_csv_fn is not None and args.val_csv_fn != '':
        small_train_dataset = dataset.PoseDataset(
            args.val_csv_fn,
            args.img_path_prefix, args.im_size,
            fliplr=False, rotate=False,
            shift=None,
            bbox_extension_range=test_bbox_extension_range,
            coord_normalize=args.coord_normalize,
            gcn=args.gcn,
            fname_index=args.fname_index,
            joint_index=args.joint_index,
            symmetric_joints=args.symmetric_joints,
            ignore_label=args.ignore_label,
            should_return_bbox=True,
            should_downscale_images=args.should_downscale_images,
            downscale_height=args.downscale_height
        )
        val_iterator = iterators.MultiprocessIterator(
            small_train_dataset, args.batch_size,
            repeat=False, shuffle=False,
            n_processes=1, n_prefetch=1)

    train_loop(net, saver, loss_op, pose_loss_op, train_op, args.dataset_name,
               train_iterator, test_iterator,
               val_iterator=val_iterator,
               max_iter=args.max_iter,
               test_step=args.test_step,
               log_step=args.log_step,
               snapshot_step=args.snapshot_step,
               batch_size=args.batch_size,
               conv_lr=args.conv_lr,
               fc_lr=args.fc_lr,
               fix_conv_iter=args.fix_conv_iter,
               output_dir=args.o_dir
               )

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
