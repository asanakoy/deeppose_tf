# Copyright (c) 2016 Artsiom Sanakoyeu

import numpy as np
"""
Person-Centric (PC) Annotations.
The canonical joint order for MPII dataset:
0 Head top
1 Neck
2 Right shoulder (from person's perspective)
3 Right elbow
4 Right wrist
5 Right hip
6 Right knee
7 Right ankle
8 Left shoulder
9 Left elbow
10 Left wrist
11 Left hip
12 Left knee
13 Left ankle
14 Thorax
15 Pelvis
"""
NUM_JOINTS = 16
CANONICAL_JOINT_NAMES = ['Head', 'Neck', 'R Shoulder',
                         'R elbow', 'R wrist',
                         'R hip', 'R knee', 'R ankle',
                         'L shoulder', 'L elbow',
                         'L wrist', 'L hip',
                         'L knee', 'L ankle',
                         'Thorax', 'Pelvis']


def joints2sticks(joints):
    """
    Args:
        joints: array of joints in the canonical order.
      The canonical joint order:
        0 Head top
        1 Neck
        2 Right shoulder (from person's perspective)
        3 Right elbow
        4 Right wrist
        5 Right hip
        6 Right knee
        7 Right ankle
        8 Left shoulder
        9 Left elbow
        10 Left wrist
        11 Left hip
        12 Left knee
        13 Left ankle
        14 Thorax
        15 Pelvis

    Returns:
        sticks: array of sticks in the canonical order.
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
    assert joints.shape == (16, 2)
    stick_n = 10  # number of stick
    sticks = np.zeros((stick_n, 4), dtype=np.float32)
    sticks[0, :] = np.hstack([joints[0, :], joints[1, :]])  # Head
    sticks[1, :] = np.hstack([joints[14, :], joints[15, :]])  # Torso
    sticks[2, :] = np.hstack([joints[2, :], joints[3, :]])  # Left U.arms
    sticks[3, :] = np.hstack([joints[3, :], joints[4, :]])  # Left L.arms
    sticks[4, :] = np.hstack([joints[5, :], joints[6, :]])  # Left U.legs
    sticks[5, :] = np.hstack([joints[6, :], joints[7, :]])  # Left L.legs
    sticks[6, :] = np.hstack([joints[8, :], joints[9, :]])  # Right U.arms
    sticks[7, :] = np.hstack([joints[9, :], joints[10, :]])  # Right L.arms
    sticks[8, :] = np.hstack([joints[11, :], joints[12, :]])  # Right U.legs
    sticks[9, :] = np.hstack([joints[12, :], joints[13, :]])  # Right L.legs
    return sticks


def convert2canonical(joints):
    """
    Convert joints to evaluation structure.
    Permute joints according to the canonical joint order.
    """
    assert joints.shape[1:] == (16, 2), 'MPII must contain 14 joints per person'
    # convert to the canonical joint order
    joint_order = [9,  # Head top
                   8,  # Neck
                   12,  # Right shoulder
                   11,  # Right elbow
                   10,  # Right wrist
                   2,  # Right hip
                   1,  # Right knee
                   0,  # Right ankle
                   13,  # Left shoulder
                   14,  # Left elbow
                   15,  # Left wrist
                   3,  # Left hip
                   4,  # Left knee
                   5,  # Left ankle
                   7,  # Thorax
                   6]  # Pelvis
    assert len(joint_order) == len(set(joint_order))
    canonical = [dict() for _ in xrange(joints.shape[0])]
    for i in xrange(joints.shape[0]):
        canonical[i]['joints'] = joints[i, joint_order, :]
        canonical[i]['sticks'] = joints2sticks(canonical[i]['joints'])
    return canonical
