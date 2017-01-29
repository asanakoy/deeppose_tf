# Based on code by Shunta Saito
# Copyright (c) 2016 Artsiom Sanakoyeu

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os.path
from config import *


def cast_path(value):
    path = None
    if value == '' or value.lower() == 'none':
        pass
    else:
        path = value
    return path


def cast_num_workers(value):
    value = int(value)
    if value < 1:
        raise ValueError('Num workers must be positive number')
    return value


def cast_downscale_height(value):
    value = int(value)
    if value < 227:
        raise ValueError('Image downscale height must be at least 227 px')
    return value


def get_arguments(argv):
    parser = argparse.ArgumentParser()

    # Basic parameters
    parser.add_argument('--max_iter', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--snapshot_step', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=1701)
    parser.add_argument('--ignore_label', type=float, default=-1)
    parser.add_argument('--dataset_name', type=str, choices=['lsp', 'mpii'], default='lsp')
    parser.add_argument(
        '--train_csv_fn', type=str,
        default=os.path.join(LSP_EXT_DATASET_ROOT, 'train_joints.csv'))
    parser.add_argument(
        '--test_csv_fn', type=str,
        default=os.path.join(LSP_EXT_DATASET_ROOT, 'test_joints.csv'))
    parser.add_argument(
        '--val_csv_fn', type=str,
        default='')
    parser.add_argument(
        '--img_path_prefix', type=str,
        default='')
    parser.add_argument('--o_dir', type=str, default=OUTPUT_DIR)
    parser.add_argument(
        '--test_step', type=int, default=100,
        help='Perform test every step iterations')
    parser.add_argument(
        '--log_step', type=int, default=1,
        help='Show loss value per this iterations')

    # Data argumentation settings
    parser.add_argument(
        '--im_size', type=int, default=227,
        help='Resize input image into this big')
    parser.add_argument(
        '--fliplr', action='store_true', default=False,
        help=('Flip image\'s left and right for data augmentation'))
    parser.add_argument(
        '--rotate', action='store_true', default=False,
        help=('Randomly rotate images for data augmentation'))
    parser.add_argument(
        '--rotate_range', type=int, default=10,
        help=('The max angle(degree) of rotation for data augmentation'))
    parser.add_argument(
        '--shift', type=float, default=0.0,
        help=('Max shift. Randomly shift bounding box for data augmentation. '
              'The value is the fraction of the bbox width and height.'))
    parser.add_argument(
        '--bbox_extension_min', type=float, default=None,
        help=('The min multiplier for joints bounding box.'))
    parser.add_argument(
        '--bbox_extension_max', type=float, default=None,
        help=('The max multiplier for joints bounding box.'))
    parser.add_argument(
        '--min_dim', type=int, default=6,
        help='Minimum dimension of a person')
    parser.add_argument(
        '--coord_normalize', action='store_true', default=True,
        help=('Perform normalization to all joint coordinates'))
    parser.add_argument(
        '--gcn', action='store_true', default=False,
        help=('Perform global contrast normalization for each input image'))

    # Data configuration
    parser.add_argument('--n_joints', type=int, default=14, help='Number of joints per person')
    parser.add_argument(
        '--fname_index', type=int, default=0,
        help='the index of image file name in a csv line')
    parser.add_argument(
        '--joint_index', type=int, default=1,
        help='the start index of joint values in a csv line')
    parser.add_argument(
        '--symmetric_joints', type=str, default='[[8, 9], [7, 10], [6, 11], [2, 3], [1, 4], [0, 5]]',
        help='Symmetric joint ids in JSON format')
    # flic_swap_joints = [(2, 4), (1, 5), (0, 6)]
    # lsp_swap_joints = [(8, 9), (7, 10), (6, 11), (2, 3), (1, 4), (0, 5)]
    # mpii_swap_joints = [(12, 13), (11, 14), (10, 15), (2, 3), (1, 4), (0, 5)]

    parser.add_argument('--should_downscale_images', action='store_true', default=False,
                        help='Downscale all images when loading to $downscale_height, rescale gt joints appropriately.')
    parser.add_argument('--downscale_height', type=cast_downscale_height, default=480,
                        help='Downscale images to this height if their height is bigger than this value. '
                             '(default=480px)')

    # Optimization settings
    parser.add_argument('--conv_lr', type=float, default=0.0005)
    parser.add_argument('--fc_lr', type=float, default=0.0005)
    parser.add_argument('--fix_conv_iter', type=int, default=0)
    parser.add_argument('--optimizer', type=str, choices=['adagrad', 'momentum', 'sgd'], default='adagrad', )
    parser.add_argument('--resume', action='store_true', default=False, help='If you want to resume training from the snapshot. '
                                                                             'Should not be used if you want to initialize only several layers from the snapshot.')
    parser.add_argument('-s', '--snapshot', type=cast_path, help='snapshot path to use as initialization or to resume training', default=None)
    parser.add_argument('--workers', type=cast_num_workers, default=1)
    parser.add_argument('--reset_iter_counter', action='store_true', default=False)
    parser.add_argument('--reset_moving_averages', action='store_true', default=False)

    parser.add_argument('--net_type', type=str, default=None,
                        help='Type of the network architecture. For ex.: Alexnet')
    args = parser.parse_args(argv)
    return args
