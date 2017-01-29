# Copyright (c) 2016 Artsiom Sanakoyeu

import numpy as np


def project_joint_onto_original_image(joints, original_bbox):
    """
    Args:
      joints: 2D array [num_joints x 2] of joints with normalized coordinates.
        Normalized coordinates (0,0) is  the center of the bbox.
        Bbox top left is (-0.5, -0.5), bottom right is (0.5, 0.5).
      original_bbox: array [x, y, w, h] - bbox coordinates (in pixels) on the original full size image.

    Returns:
      projected_joints: in pixel coordinates
    """
    if joints.ndim != 2 or joints.shape[1] != 2:
        raise ValueError('joints must be 2D array [num_joints x 2]')
    if joints.min() < -0.501 or joints.max() > 0.501:
        raise ValueError(
            'Joints\' coordinates must be normalized and be in [-0.5, 0.5], got[{}, {}]'.format(
                joints.min(), joints.max()))
    original_bbox = original_bbox.astype(int)
    x, y, w, h = original_bbox
    projected_joints = np.array(joints, dtype=np.float32)
    projected_joints += np.array([0.5, 0.5])
    projected_joints[:, 0] *= w
    projected_joints[:, 1] *= h
    projected_joints += np.array([x, y])
    return projected_joints
