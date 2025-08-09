import csv
import numpy as np
import utils.pc_utils as pc_utils
from typing import List

import torch
from pytorch3d.ops import box3d_overlap


def decode_angles(angles, num_heading_areas):
    """
    Convert angles to heading_area_ids and heading offsets
    Area centers at 0, 1 * (2pi / N), ..., (N - 1) * (2pi / N)
    :param angles: heading angles
    :param num_heading_areas: number of heading areas
    :return: area_ids: 0, 1,..., num_heading_areas - 1
    :return: offsets: area_id * (2pi / num_heading_areas) + offsets = angle
    """
    heading_area_size = 2 * np.pi / float(num_heading_areas)
    shifted_angles = (angles + heading_area_size / 2) % (2 * np.pi)
    if torch.is_tensor(shifted_angles):
        area_ids = (shifted_angles / heading_area_size).long()
    else:
        area_ids = (shifted_angles / heading_area_size).astype(np.int64)
    offsets = shifted_angles - (area_ids * heading_area_size + heading_area_size / 2)
    return area_ids, offsets


def encode_angles(area_ids, offsets, num_heading_areas):
    """
    Inverse function to decode_angles
    :param area_ids: 0, 1,..., num_heading_areas - 1
    :param offsets: area_id * (2pi / num_heading_areas) + offsets = angles
    :param num_heading_areas: number of heading areas
    :return: angles: 0 ~ 2pi
    """
    heading_area_size = 2 * np.pi / float(num_heading_areas)
    angle_centers = area_ids * heading_area_size
    angles = angle_centers + offsets
    if torch.is_tensor(angles):
        convert_value = (angles > 2 * torch.pi).float() * 2 * torch.pi
    else:
        convert_value = (angles > 2 * np.pi).astype(np.float32) * 2 * np.pi
    return angles - convert_value


def boxes_to_corners_3d(centers, angles, sizes):
    """
    Convert 3D bounding box size and angle to 3D bounding box corners
    Corner point order: (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)
    :param centers: shape(..., 3), box centers
    :param angles: shape(...), heading angles
    :param sizes: shape(..., 3), box sizes
    :return: corners_3d: shape(..., 8, 3), 3D box corners
    """
    assert centers.shape[: -1] == angles.shape == sizes.shape[: -1]
    assert centers.shape[-1] == sizes.shape[-1] == 3
    shapes = centers.shape[: -1]
    centers, angles, sizes = centers.view(-1, 3), angles.view(-1), sizes.view(-1, 3)
    batch_size, device = centers.shape[0], centers.device
    # Rotation matrix
    cos_a, sin_a = torch.cos(angles), torch.sin(angles)
    rot = torch.zeros((batch_size, 3, 3), device=device)
    rot[:, 0, 0], rot[:, 0, 1], rot[:, 1, 0], rot[:, 1, 1], rot[:, 2, 2] = cos_a, -sin_a, sin_a, cos_a, 1
    # Relative positions of corners
    # Orders: (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)
    x_corners = sizes[:, [0 for _ in range(8)]] / 2 * torch.tensor([-1, 1, 1, -1, -1, 1, 1, -1], device=device)
    y_corners = sizes[:, [1 for _ in range(8)]] / 2 * torch.tensor([-1, -1, 1, 1, -1, -1, 1, 1], device=device)
    z_corners = sizes[:, [2 for _ in range(8)]] / 2 * torch.tensor([-1, -1, -1, -1, 1, 1, 1, 1], device=device)
    corners_3d = torch.bmm(rot, torch.stack([x_corners, y_corners, z_corners], dim=-2)) + centers.unsqueeze(-1)
    corners_3d = corners_3d.transpose(-1, -2)
    return corners_3d.view(shapes + (8, 3))


def boxes_to_corners_3d_np(center, angle, size):
    """
    Convert a single 3D bounding box size and angle to 3D bounding box corners
    Corner point order: (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)
    :param center: shape (3,), x, y, z
    :param angle: shape (1,), heading_angle 0 ~ 2pi
    :param size: shape (3,), l, w, h
    :return: corners_3d: shape (8, 3), 3D box corners
    """
    r = pc_utils.rot_z(angle)
    l, w, h = size / 2
    x_corners = [-l, l, l, -l, -l, l, l, -l]
    y_corners = [-w, -w, w, w, -w, -w, w, w]
    z_corners = [-h, -h, -h, -h, h, h, h, h]
    corners_3d = np.dot(r, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] += center[0]
    corners_3d[1, :] += center[1]
    corners_3d[2, :] += center[2]
    return np.transpose(corners_3d)


def points_in_boxes_3d_aligned(points, corners_3d):
    """
    Return whether the points are inside the boxes, where boxes are aligned with the axes.
    :param points: shape: (batch_size, num_points, 3)
    :param corners_3d: shape: (batch_size, num_proposals, 8, 3)
    :return: points_in: shape: (batch_size, num_proposals, num_points), 1 for points that are in the box
    """
    min_coords, max_coords = torch.min(corners_3d, dim=-2)[0], torch.max(corners_3d, dim=-2)[0]
    min_coords, max_coords = min_coords.unsqueeze(-2), max_coords.unsqueeze(-2)
    points_xyz = points[:, :, : 3].unsqueeze(1)
    points_in = torch.gt(points_xyz, min_coords) * torch.lt(points_xyz, max_coords)
    points_in = points_in[:, :, :, 0] * points_in[:, :, :, 1] * points_in[:, :, :, 2]
    return points_in


def points_in_boxes_3d(point_clouds, corners_3d):
    """
    Return whether the points are inside the boxes
    :param point_clouds: shape: (batch_size, num_points, 3)
    :param corners_3d: shape: (batch_size, num_proposals, 8, 3)
    :return: points_in: shape: (batch_size, num_proposals, num_points), 1 for points that are in the box
    """
    batch_size, num_proposals, num_points = corners_3d.shape[0], corners_3d.shape[1], point_clouds.shape[1]
    vectors_xy = corners_3d[:, :, 1, : 2] - corners_3d[:, :, 0, : 2]
    vectors_d = torch.sqrt(torch.sum(torch.square(vectors_xy), dim=-1))
    angles_sin = - vectors_xy[:, :, 1] / vectors_d  # rotate -angle to get the axis-aligned box
    angles_cos = vectors_xy[:, :, 0] / vectors_d
    rot_matrix = torch.stack(
        [angles_cos, -angles_sin, torch.zeros_like(angles_sin),
         angles_sin, angles_cos, torch.zeros_like(angles_sin),
         torch.zeros_like(angles_sin), torch.zeros_like(angles_sin), torch.ones_like(angles_sin)],
        dim=-1
    ).view(batch_size, num_proposals, 3, 3)
    points_in_list = []
    for matrix, points, corners in zip(rot_matrix, point_clouds[:, :, : 3], corners_3d):
        rotated_pc = torch.matmul(matrix.unsqueeze(1), points.unsqueeze(-1)).squeeze(-1)
        rotated_corners = torch.matmul(matrix, corners.transpose(-1, -2)).transpose(-1, -2)
        min_coords, max_coords = torch.min(rotated_corners, dim=-2)[0], torch.max(rotated_corners, dim=-2)[0]
        min_coords, max_coords = min_coords.unsqueeze(-2), max_coords.unsqueeze(-2)
        points_in = torch.gt(rotated_pc, min_coords) * torch.lt(rotated_pc, max_coords)
        points_in = points_in[:, :, 0] * points_in[:, :, 1] * points_in[:, :, 2]
        points_in_list.append(points_in)
    points_in = torch.stack(points_in_list, dim=0)
    return points_in


def remove_empty_box(corners_3d, point_clouds, axis_aligned=False):
    """
    Remove predicted boxes that contain no points (point number less than 5).
    :param corners_3d: shape: (batch_size, num_proposals, 8, 3)
    :param point_clouds: shape: (batch_size, num_points, 3)
    :param axis_aligned: whether the boxes are aligned with the axes
    :return: nonempty_box_mask: shape: (batch_size, num_proposals), 1 for proposals that are not empty
    :return: point_in: shape: (batch_size, num_proposals, num_points), 1 for points that are in the box
    """
    if axis_aligned:
        points_in = points_in_boxes_3d_aligned(point_clouds, corners_3d)
    else:
        points_in = points_in_boxes_3d(point_clouds, corners_3d)
    points_inbox_num = torch.sum(points_in, dim=-1)
    non_empty_box_mask = torch.gt(points_inbox_num, 5)
    return non_empty_box_mask, points_in


def nms_3d(corners_3d, obj_prob, sem_cls, label_mask=None, nms_threshold=0.25):
    """
    Non-maximum suppression for 3D boxes.
    If the 3D-IoU of two boxes is above nms_threshold, and when the semantic class of the two boxes are the same,
    suppress the one with lower obj_prob.
    :param corners_3d: shape (batch_size, num_proposals, 8, 3)
    :param obj_prob: shape (batch_size, num_proposals)
    :param sem_cls: shape (batch_size, num_proposals)
    :param label_mask: shape (batch_size, num_proposals), 1 for proposals that are not empty
    :param nms_threshold: threshold for 3D-IoU
    :return: label_mask: shape (batch_size, num_proposals), 1 for proposals that are not suppressed
    """
    batch_size, num_proposals = corners_3d.shape[0], corners_3d.shape[1]
    if label_mask is None:
        label_mask = torch.ones_like(obj_prob, dtype=torch.bool)
    assert obj_prob.shape == sem_cls.shape == label_mask.shape == (batch_size, num_proposals)
    valid_batch_ids, valid_box_ids = torch.where(label_mask)
    for i, boxes in enumerate(corners_3d):
        # Select valid boxes, sort by obj_prob
        valid_box_id = valid_box_ids[valid_batch_ids == i]
        valid_box_id = valid_box_id[torch.argsort(obj_prob[i, valid_box_id], descending=True)]
        valid_boxes = boxes[valid_box_id, :, :]
        # Calculate 3d-IoUs between all valid boxes
        try:
            _, ious = box3d_overlap(valid_boxes, valid_boxes)
        except (ValueError, RuntimeError):
            continue
        # Keep the upper triangular part of the IoU matrix
        ious = torch.triu(ious, diagonal=1)
        # Ignore the boxes whose semantic class is different
        ious *= torch.eq(sem_cls[i, valid_box_id].unsqueeze(0), sem_cls[i, valid_box_id].unsqueeze(1)).float()
        pair_id1, pair_id2 = torch.where(torch.gt(ious, nms_threshold))
        pair_id1, pair_id2 = pair_id1.detach().cpu().numpy(), pair_id2.detach().cpu().numpy()
        suppress = set()
        for element1, element2 in zip(pair_id1, pair_id2):
            if element1 in suppress:
                continue
            suppress.add(element2)
        label_mask[i, valid_box_id[list(suppress)]] = False
    return label_mask


def enclosing_box3d_vol(corners1, corners2):
    """
    volume of enclosing axis-aligned box
    """
    assert len(corners1.shape) == 4
    assert len(corners2.shape) == 4
    assert corners1.shape[0] == corners2.shape[0]
    assert corners1.shape[2] == 8
    assert corners1.shape[3] == 3
    assert corners2.shape[2] == 8
    assert corners2.shape[3] == 3
    corners1 = corners1.clone()
    corners2 = corners2.clone()
    # flip Y axis, since it is negative
    corners1[:, :, :, 1] *= -1
    corners2[:, :, :, 1] *= -1
    al_xmin = torch.min(
        torch.min(corners1[:, :, :, 0], dim=2).values[:, :, None],
        torch.min(corners2[:, :, :, 0], dim=2).values[:, None, :],
    )
    al_ymin = torch.max(
        torch.max(corners1[:, :, :, 1], dim=2).values[:, :, None],
        torch.max(corners2[:, :, :, 1], dim=2).values[:, None, :],
    )
    al_zmin = torch.min(
        torch.min(corners1[:, :, :, 2], dim=2).values[:, :, None],
        torch.min(corners2[:, :, :, 2], dim=2).values[:, None, :],
    )
    al_xmax = torch.max(
        torch.max(corners1[:, :, :, 0], dim=2).values[:, :, None],
        torch.max(corners2[:, :, :, 0], dim=2).values[:, None, :],
    )
    al_ymax = torch.min(
        torch.min(corners1[:, :, :, 1], dim=2).values[:, :, None],
        torch.min(corners2[:, :, :, 1], dim=2).values[:, None, :],
    )
    al_zmax = torch.max(
        torch.max(corners1[:, :, :, 2], dim=2).values[:, :, None],
        torch.max(corners2[:, :, :, 2], dim=2).values[:, None, :],
    )
    diff_x = torch.abs(al_xmax - al_xmin)
    diff_y = torch.abs(al_ymax - al_ymin)
    diff_z = torch.abs(al_zmax - al_zmin)
    vol = diff_x * diff_y * diff_z
    return vol


def box3d_vol_tensor(corners):
    EPS = 1e-6
    reshape = False
    B, K = corners.shape[0], corners.shape[1]
    if len(corners.shape) == 4:
        # batch x prop x 8 x 3
        reshape = True
        corners = corners.view(-1, 8, 3)
    a = torch.sqrt(
        (corners[:, 0, :] - corners[:, 1, :]).pow(2).sum(dim=1).clamp(min=EPS)
    )
    b = torch.sqrt(
        (corners[:, 1, :] - corners[:, 2, :]).pow(2).sum(dim=1).clamp(min=EPS)
    )
    c = torch.sqrt(
        (corners[:, 0, :] - corners[:, 4, :]).pow(2).sum(dim=1).clamp(min=EPS)
    )
    vols = a * b * c
    if reshape:
        vols = vols.view(B, K)
    return vols


@torch.jit.ignore
def to_list_1d(arr) -> List[float]:
    arr = arr.detach().cpu().numpy().tolist()
    return arr


@torch.jit.ignore
def to_list_3d(arr) -> List[List[List[float]]]:
    arr = arr.detach().cpu().numpy().tolist()
    return arr


def helper_computeIntersection(
    cp1: torch.Tensor, cp2: torch.Tensor, s: torch.Tensor, e: torch.Tensor
):
    dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
    dp = [s[0] - e[0], s[1] - e[1]]
    n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
    n2 = s[0] * e[1] - s[1] * e[0]
    n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
    # return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]
    return torch.stack([(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3])


def helper_inside(cp1: torch.Tensor, cp2: torch.Tensor, p: torch.Tensor):
    ineq = (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])
    return ineq.item()


def polygon_clip_unnest(subjectPolygon: torch.Tensor, clipPolygon: torch.Tensor):
    """Clip a polygon with another polygon.

    Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python

    Args:
      subjectPolygon: a list of (x,y) 2d points, any polygon.
      clipPolygon: a list of (x,y) 2d points, has to be *convex*
    Note:
      **points have to be counter-clockwise ordered**

    Return:
      a list of (x,y) vertex point for the intersection polygon.
    """
    outputList = [subjectPolygon[x] for x in range(subjectPolygon.shape[0])]
    cp1 = clipPolygon[-1]
    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList.copy()
        outputList.clear()
        s = inputList[-1]
        for subjectVertex in inputList:
            e = subjectVertex
            if helper_inside(cp1, cp2, e):
                if not helper_inside(cp1, cp2, s):
                    outputList.append(helper_computeIntersection(cp1, cp2, s, e))
                outputList.append(e)
            elif helper_inside(cp1, cp2, s):
                outputList.append(helper_computeIntersection(cp1, cp2, s, e))
            s = e
        cp1 = cp2
        if len(outputList) == 0:
            # return None
            break
    return outputList


def generalized_box3d_iou_tensor(
        corners1: torch.Tensor,
        corners2: torch.Tensor,
        nums_k2: torch.Tensor,
        rotated_boxes: bool = True,
        return_inter_vols_only: bool = False,
):
    """
    Input:
        corners1: torch Tensor (B, K1, 8, 3), assume up direction is negative Y
        corners2: torch Tensor (B, K2, 8, 3), assume up direction is negative Y
        Assumes that the box is only rotated along Z direction
    Returns:
        B x K1 x K2 matrix of generalized IOU by approximating the boxes to be axis aligned
    """
    assert len(corners1.shape) == 4
    assert len(corners2.shape) == 4
    assert corners1.shape[2] == 8
    assert corners1.shape[3] == 3
    assert corners1.shape[0] == corners2.shape[0]
    assert corners1.shape[2] == corners2.shape[2]
    assert corners1.shape[3] == corners2.shape[3]
    B, K1 = corners1.shape[0], corners1.shape[1]
    _, K2 = corners2.shape[0], corners2.shape[1]
    # box height. Y is negative, so max is torch.min
    ymax = torch.min(corners1[:, :, 0, 1][:, :, None], corners2[:, :, 0, 1][:, None, :])
    ymin = torch.max(corners1[:, :, 4, 1][:, :, None], corners2[:, :, 4, 1][:, None, :])
    height = (ymax - ymin).clamp(min=0)
    EPS = 1e-8
    idx = torch.arange(start=3, end=-1, step=-1, device=corners1.device)
    idx2 = torch.tensor([0, 2], dtype=torch.int64, device=corners1.device)
    rect1 = corners1[:, :, idx, :]
    rect2 = corners2[:, :, idx, :]
    rect1 = rect1[:, :, :, idx2]
    rect2 = rect2[:, :, :, idx2]
    lt = torch.max(rect1[:, :, 1][:, :, None, :], rect2[:, :, 1][:, None, :, :])
    rb = torch.min(rect1[:, :, 3][:, :, None, :], rect2[:, :, 3][:, None, :, :])
    wh = (rb - lt).clamp(min=0)
    non_rot_inter_areas = wh[:, :, :, 0] * wh[:, :, :, 1]
    non_rot_inter_areas = non_rot_inter_areas.view(B, K1, K2)
    if nums_k2 is not None:
        for b in range(B):
            non_rot_inter_areas[b, :, nums_k2[b]:] = 0
    enclosing_vols = enclosing_box3d_vol(corners1, corners2)
    # vols of boxes
    vols1 = box3d_vol_tensor(corners1).clamp(min=EPS)
    vols2 = box3d_vol_tensor(corners2).clamp(min=EPS)
    sum_vols = vols1[:, :, None] + vols2[:, None, :]
    # filter malformed boxes
    good_boxes = (enclosing_vols > 2 * EPS) * (sum_vols > 4 * EPS)
    if rotated_boxes:
        inter_areas = torch.zeros((B, K1, K2), dtype=torch.float32)
        rect1 = rect1.cpu()
        rect2 = rect2.cpu()
        nums_k2_np = to_list_1d(nums_k2)
        non_rot_inter_areas_np = to_list_3d(non_rot_inter_areas)
        for b in range(B):
            for k1 in range(K1):
                for k2 in range(K2):
                    if nums_k2 is not None and k2 >= nums_k2_np[b]:
                        break
                    if non_rot_inter_areas_np[b][k1][k2] == 0:
                        continue
                    # compute volume of intersection
                    inter = polygon_clip_unnest(rect1[b, k1], rect2[b, k2])
                    if len(inter) > 0:
                        xs = torch.stack([x[0] for x in inter])
                        ys = torch.stack([x[1] for x in inter])
                        inter_areas[b, k1, k2] = torch.abs(
                            torch.dot(xs, torch.roll(ys, 1))
                            - torch.dot(ys, torch.roll(xs, 1))
                        )
        inter_areas.mul_(0.5)
    else:
        inter_areas = non_rot_inter_areas
    inter_areas = inter_areas.to(corners1.device)
    # gIOU = iou - (1 - sum_vols/enclose_vol)
    inter_vols = inter_areas * height
    if return_inter_vols_only:
        return inter_vols
    union_vols = (sum_vols - inter_vols).clamp(min=EPS)
    ious = inter_vols / union_vols
    giou_second_term = -(1 - union_vols / enclosing_vols)
    gious = ious + giou_second_term
    gious *= good_boxes
    if nums_k2 is not None:
        mask = torch.zeros((B, K1, K2), device=height.device, dtype=torch.float32)
        for b in range(B):
            mask[b, :, : nums_k2[b]] = 1
        gious *= mask
    return gious


generalized_box3d_iou_tensor_jit = torch.jit.script(generalized_box3d_iou_tensor)


def generalized_box3d_iou(
        corners1: torch.Tensor,
        corners2: torch.Tensor,
        nums_k2: torch.Tensor,
        rotated_boxes: bool = True,
        return_inter_vols_only: bool = False,
        needs_grad: bool = False,
):
    context = torch.enable_grad if needs_grad else torch.no_grad
    with context():
        return generalized_box3d_iou_tensor_jit(
            corners1, corners2, nums_k2, rotated_boxes, return_inter_vols_only
        )
