# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Utility functions for processing point clouds.

# Author: Charles R. Qi and Or Litany

# Updated by Zijing Zhao (Peking University)
# Date: Oct 2023
# Note: simplify the file


import numpy as np
import torch


def random_sampling(pc, num_sample, seed=None, replace=None, return_choices=False):
    """
    Input is N x C, output is num_sample x C
    """
    if replace is None:
        replace = (pc.shape[0] < num_sample)
    if seed is not None:
        random_state = np.random.RandomState(seed)
        choices = random_state.choice(pc.shape[0], num_sample, replace=replace)
    else:
        choices = np.random.choice(pc.shape[0], num_sample, replace=replace)
    if return_choices:
        return pc[choices], choices
    else:
        return pc[choices]


def rot_z(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


def extract_pc_in_box3d(pc, corners_3d):
    vector_xy = corners_3d[1, : 2] - corners_3d[0, : 2]
    vector_d = np.sqrt(np.sum(np.square(vector_xy)))
    if vector_d <= 1e-4:
        raise ValueError('Invalid box!')
    angle_sin = - vector_xy[1] / vector_d  # rotate -angle to get the axis-aligned box
    angle_cos = vector_xy[0] / vector_d
    rot_matrix = np.array(
        [[angle_cos, -angle_sin, 0.0],
         [angle_sin, angle_cos, 0.0],
         [0.0, 0.0, 1.0]]
    )
    pc_rot = np.matmul(rot_matrix, pc[:, : 3].transpose(1, 0)).transpose(1, 0)
    corners_rot = np.matmul(rot_matrix, corners_3d.transpose(1, 0)).transpose(1, 0)
    return extract_pc_in_box3d_aligned(pc_rot, corners_rot)


def extract_pc_in_box3d_aligned(pc, corners_3d):
    min_coords, max_coords = np.min(corners_3d, axis=0), np.max(corners_3d, axis=0)
    pc_in_box = np.logical_and(
        np.logical_and(pc[:, 0] >= min_coords[0], pc[:, 0] <= max_coords[0]),
        np.logical_and(pc[:, 1] >= min_coords[1], pc[:, 1] <= max_coords[1])
    )
    pc_in_box = np.logical_and(pc_in_box, np.logical_and(pc[:, 2] >= min_coords[2], pc[:, 2] <= max_coords[2]))
    return pc[pc_in_box, :], pc_in_box


def shift_scale_points(pred_xyz, src_range, dst_range=None):
    """
    pred_xyz: B x N x 3
    src_range: [[B x 3], [B x 3]] - min and max XYZ coords
    dst_range: [[B x 3], [B x 3]] - min and max XYZ coords
    """
    if dst_range is None:
        dst_range = [
            torch.zeros((src_range[0].shape[0], 3), device=src_range[0].device),
            torch.ones((src_range[0].shape[0], 3), device=src_range[0].device),
        ]

    if pred_xyz.ndim == 4:
        src_range = [x[:, None] for x in src_range]
        dst_range = [x[:, None] for x in dst_range]

    assert src_range[0].shape[0] == pred_xyz.shape[0]
    assert dst_range[0].shape[0] == pred_xyz.shape[0]
    assert src_range[0].shape[-1] == pred_xyz.shape[-1]
    assert src_range[0].shape == src_range[1].shape
    assert dst_range[0].shape == dst_range[1].shape
    assert src_range[0].shape == dst_range[1].shape

    src_diff = src_range[1][:, None, :] - src_range[0][:, None, :]
    dst_diff = dst_range[1][:, None, :] - dst_range[0][:, None, :]
    prop_xyz = (
        ((pred_xyz - src_range[0][:, None, :]) * dst_diff) / src_diff
    ) + dst_range[0][:, None, :]
    return prop_xyz


def scale_points(pred_xyz, mult_factor):
    if pred_xyz.ndim == 4:
        mult_factor = mult_factor[:, None]
    scaled_xyz = pred_xyz * mult_factor[:, None, :]
    return scaled_xyz
