# Copyright (c) 2016 Artsiom Sanakoyeu

import numpy as np
import poseevaluation
"""
The canonical part stick order:
0 Head
1 Torso
2 Right Upper Arm
3 Right Lower Arm
4 Right Upper Leg
5 Right Lower Leg
6 Left Upper Arm
7 Left Lower Arm
8 Left Upper Leg
9 Left Lower Leg
"""

CANONICAL_STICK_NAMES = ['Head', 'Torso', 'RU Arm', 'RL Arm', 'RU Leg',
                         'RL Leg', 'LU Arm', 'LL Arm', 'LU Leg', 'LL Leg']


def eval_relaxed_pcp(gt_joints, predicted_joints, thresh=0.5):
    """
    Relaxed PCP as in DeepPose paper.
    Compute average relaxed pcp per stick.
    Args:
      gt_joints, predicted_joints: arrays of gt and predicted joints in the canonical order
      thresh: fraction of the gt stick length. This is the maximal average deviation of the
        predicted joints of the stick from the gt joints position of the stick.
    Returns:
        pcp_per_stick: array of pcp scores. i-th element is the pcp score for the i-th stick
    """
    if len(gt_joints) != len(predicted_joints):
        raise ValueError('Len of gt must be equal to len of predicted')
    if len(gt_joints) == 0:
        raise ValueError('Empty array')

    num_examples = len(gt_joints)
    # the number of sticks for a pose
    num_sticks = gt_joints[0]['sticks'].shape[0]
    if num_sticks != 10:
        raise ValueError('PCP requires 10 sticks. Provided: {}'.format(num_sticks))
    is_matched = np.zeros((num_examples, num_sticks), dtype=int)

    for i in xrange(num_examples):
        for stick_id in xrange(num_sticks):
            gt_stick_len = np.linalg.norm(gt_joints[i]['sticks'][stick_id, :2] -
                                          gt_joints[i]['sticks'][stick_id, 2:])
            delta_a = np.linalg.norm(predicted_joints[i]['sticks'][stick_id, :2] -
                                     gt_joints[i]['sticks'][stick_id, :2]) / gt_stick_len
            delta_b = np.linalg.norm(predicted_joints[i]['sticks'][stick_id, 2:] -
                                     gt_joints[i]['sticks'][stick_id, 2:]) / gt_stick_len
            delta = (delta_a + delta_b) / 2.0

            is_matched[i, stick_id] = delta <= thresh
    pcp_per_stick = np.mean(is_matched, 0)
    return pcp_per_stick


def eval_strict_pcp(gt_joints, predicted_joints, thresh=0.5):
    """
    Compute average pcp per stick
    Args:
      gt_joints, predicted_joints: arrays of gt and predicted joints in the canonical order
      thresh: fraction of the gt stick length. This is the maximal deviation of the
        predicted joint from the gt joint position.
    Returns:
        pcp_per_stick: array of pcp scores. i-th element is the pcp score for the i-th stick
    """
    if len(gt_joints) != len(predicted_joints):
        raise ValueError('Len of gt must be equal to len of predicted')
    if len(gt_joints) == 0:
        raise ValueError('Empty array')

    num_examples = len(gt_joints)
    # the number of sticks for a pose
    num_sticks = gt_joints[0]['sticks'].shape[0]
    if num_sticks != 10:
        raise ValueError('PCP requires 10 sticks. Provided: {}'.format(num_sticks))
    is_matched = np.zeros((num_examples, num_sticks), dtype=int)

    for i in xrange(num_examples):
        for stick_id in xrange(num_sticks):
            gt_stick_len = np.linalg.norm(gt_joints[i]['sticks'][stick_id, :2] -
                                          gt_joints[i]['sticks'][stick_id, 2:])
            delta_a = np.linalg.norm(predicted_joints[i]['sticks'][stick_id, :2] -
                                            gt_joints[i]['sticks'][stick_id, :2]) / gt_stick_len
            delta_b = np.linalg.norm(predicted_joints[i]['sticks'][stick_id, 2:] -
                                            gt_joints[i]['sticks'][stick_id, 2:]) / gt_stick_len

            is_matched[i, stick_id] = (delta_a <= thresh and delta_b <= thresh)
    pcp_per_stick = np.mean(is_matched, 0)
    return pcp_per_stick


def average_pcp_left_right_limbs(pcp_per_stick):
    part_names = ['Head', 'Torso', 'U Arm', 'L Arm', 'U Leg', 'L Leg', 'mean']
    pcp_per_part = pcp_per_stick[:2].tolist() + \
                   [(pcp_per_stick[i] + pcp_per_stick[i + 4]) / 2 for i in xrange(2, 6)]
    pcp_per_part.append(np.mean(pcp_per_part))
    return pcp_per_part, part_names


def eval_pckh(dataset_name, gt_joints, predicted_joints, thresh=0.5):
    """
    Compute average PCKh per joint.
    Matching threshold is 50% (thresh) of the head segment box size by default
    Args:
      gt_joints, predicted_joints: arrays of gt and predicted joints in the canonical order
      thresh: fraction of the head segment length. This is the maximal deviation of the
        predicted joint from the gt joint position.
    Returns:
        pckh_per_joint: array of PCKh scores. i-th element is the PCKh score for the i-th joint
    """
    if len(gt_joints) != len(predicted_joints):
        raise ValueError('Len of gt must be equal to len of predicted')
    if len(gt_joints) == 0:
        raise ValueError('Empty array')
    num_joints = poseevaluation.__dict__[dataset_name].NUM_JOINTS
    num_examples = len(gt_joints)

    is_matched = np.zeros((num_examples, num_joints), dtype=int)

    for i in xrange(num_examples):
        if gt_joints[i]['joints'].shape != (num_joints, 2):
            raise ValueError('MPII::PCKh requires 16 joints with 2D coordinates for each.'
                             ' Person {}: provided joints shape: {}'.format(i, gt_joints[0]['joints'].shape))
        head_id = 0
        gt_head_len = np.linalg.norm(gt_joints[i]['sticks'][head_id, :2] -
                                     gt_joints[i]['sticks'][head_id, 2:])
        for joint_id in xrange(num_joints):
            delta = np.linalg.norm(predicted_joints[i]['joints'][joint_id] -
                                   gt_joints[i]['joints'][joint_id]) / gt_head_len

            is_matched[i, joint_id] = delta <= thresh
    pckh_per_joint = np.mean(is_matched, 0)
    return pckh_per_joint


def average_pckh_symmetric_joints(dataset_name, pckh_per_joint):
    if dataset_name not in ['mpii', 'lsp']:
        raise ValueError('Unknown dataset {}'.format(dataset_name))

    joint_names = ['Head', 'Neck', 'Shoulder',
                   'Elbow', 'Wrist',
                   'Hip', 'Knee', 'Ankle',
                   'Thorax', 'Pelvis']
    if dataset_name == 'lsp':
        joint_names = joint_names[:-2]
    pckh_symmetric_joints = pckh_per_joint[:2].tolist()
    for i in xrange(2, 8):
        pckh_symmetric_joints.append((pckh_per_joint[i] + pckh_per_joint[i + 6]) / 2.0)
    pckh_symmetric_joints += pckh_per_joint[14:].tolist()
    return pckh_symmetric_joints, joint_names
