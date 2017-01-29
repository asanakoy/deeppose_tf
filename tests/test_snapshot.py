#!/usr/bin/env python
# Copyright (c) 2016 Artsiom Sanakoyeu

import numpy as np
import copy
import os
import sys
import math
from tqdm import tqdm
from scripts import dataset
import scripts.regressionnet
from scripts.regressionnet import batch2feeds, calculate_metric
from chainer import iterators

from scripts import config


def get_gt_data(test_iterator):
    test_it = copy.copy(test_iterator)
    num_test_examples = len(test_it.dataset)
    num_batches = int(math.ceil(num_test_examples / test_it.batch_size))
    gt_joints = list()
    gt_joints_is_valid = list()
    orig_bboxes = list()

    print len(test_it.dataset)
    for i, batch in tqdm(enumerate(test_it), total=num_batches):
        feeds = batch2feeds(batch)
        gt_joints.append(feeds[1])
        gt_joints_is_valid.append(feeds[2])
        orig_bboxes.append(np.vstack([x['bbox'] for x in feeds[3]]))

    gt_joints = np.vstack(gt_joints)
    gt_joints_is_valid = np.vstack(gt_joints_is_valid)
    orig_bboxes = np.vstack(orig_bboxes)
    return gt_joints, gt_joints_is_valid, orig_bboxes


def main(dataset_name, snapshot_path):
    """
    Args:
        dataset_name: 'mpii' or 'lsp'.
        init_snapshot_path: path to the snapshot to test
    """
    if dataset_name == 'mpii':
        TEST_CV_FILEPATH = os.path.join(config.ROOT_DIR, 'datasets/mpii/test_joints.csv')
        IMG_PATH_PREFIX = os.path.join(config.ROOT_DIR, 'datasets/mpii/images')
        symmetric_joints = "[[12, 13], [11, 14], [10, 15], [2, 3], [1, 4], [0, 5]]"
        ignore_label = -100500
    else:
        TEST_CV_FILEPATH = os.path.join(config.ROOT_DIR, 'datasets/lsp_ext/test_joints.csv')
        IMG_PATH_PREFIX = ''
        symmetric_joints = "[[8, 9], [7, 10], [6, 11], [2, 3], [1, 4], [0, 5]]"
        ignore_label = -1

    test_dataset = dataset.PoseDataset(
        TEST_CV_FILEPATH,
        IMG_PATH_PREFIX, 227,
        fliplr=False, rotate=False,
        shift=None,
        bbox_extension_range=(1.0, 1.0),
        coord_normalize=True,
        gcn=True,
        fname_index=0,
        joint_index=1,
        symmetric_joints=symmetric_joints,
        ignore_label=ignore_label,
        should_return_bbox=True,
        should_downscale_images=True,
        downscale_height=400
    )

    test_iterator = iterators.MultiprocessIterator(
        test_dataset, batch_size=128,
        repeat=False, shuffle=False,
        n_processes=1, n_prefetch=1)

    test_net(test_dataset, test_iterator, dataset_name, snapshot_path)


def test_net(test_dataset, test_iterator, dataset_name, snapshot_path):
    if dataset_name not in ['lsp', 'mpii']:
        raise ValueError('Unknown dataset: {}'.format(dataset_name))

    net, loss_op, pose_loss_op, train_op = scripts.regressionnet.create_regression_net(
        n_joints=16 if dataset_name == 'mpii' else 14,
        init_snapshot_path=snapshot_path,
        is_resume=True,
        net_type='Alexnet',
        optimizer_type='momentum',
        gpu_memory_fraction=0.32)  # Set how much GPU memory to reserve for the network
    print snapshot_path
    for ext in np.linspace(1.0, 2.0, 6, True):
        print '\n===================='
        print 'BBOX EXTENSION:', ext
        test_dataset.bbox_extension_range = (ext, ext)
        scripts.regressionnet.evaluate_pcp(net, pose_loss_op, test_iterator, None,
                                           dataset_name=dataset_name,
                                           tag_prefix='test')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'Wrong arguments passed.'
        print 'USAGE: {} (mpii|lsp) snapshot_path'
    else:
        dataset_name = sys.argv[1]
        snapshot_path = sys.argv[2]

        # dataset_name = 'mpii'
        # init_snapshot_path = os.path.join(config.ROOT_DIR, 'out/mpii_alexnet_imagenet/checkpoint-10000')
        # init_snapshot_path = os.path.join(config.ROOT_DIR, 'out/lsp_alexnet_scratch/checkpoint-10000')
        main(dataset_name, snapshot_path)
