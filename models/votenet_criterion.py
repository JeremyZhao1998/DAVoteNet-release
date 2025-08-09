# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Updated by Zijing Zhao (Peking University)
# Date: Oct 2023

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy, binary_cross_entropy_with_logits


def huber_loss(error, delta=1.0):
    """
    Args:
        error: Torch tensor (d1,d2,...,dk)
        delta: float
    Returns:
        loss: Torch tensor (d1,d2,...,dk)

    x = error = pred - gt or dist(pred,gt)
    0.5 * |x|^2                 if |x|<=d
    0.5 * d^2 + d * (|x|-d)     if |x|>d
    Ref: https://github.com/charlesq34/frustum-pointnets/blob/master/models/model_util.py
    """
    abs_error = torch.abs(error)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = (abs_error - quadratic)
    loss = 0.5 * quadratic ** 2 + delta * linear
    return loss


def nn_distance(pc1, pc2, l1_smooth=False, delta=1.0, l1=False):
    """
    Input:
        pc1: (B,n,C) torch tensor
        pc2: (B,m,C) torch tensor
        l1_smooth: bool, whether to use l1_smooth loss
        delta: scalar, the delta used in l1_smooth loss
    Output:
        dist1: (B,n) torch float32 tensor
        idx1: (B,n) torch int64 tensor
        dist2: (B,m) torch float32 tensor
        idx2: (B,m) torch int64 tensor
    """
    n = pc1.shape[1]
    m = pc2.shape[1]
    pc1_expand_tile = pc1.unsqueeze(2).repeat(1, 1, m, 1)
    pc2_expand_tile = pc2.unsqueeze(1).repeat(1, n, 1, 1)
    pc_diff = pc1_expand_tile - pc2_expand_tile
    if l1_smooth:
        pc_dist = torch.sum(huber_loss(pc_diff, delta), dim=-1)  # (B, n, m)
    elif l1:
        pc_dist = torch.sum(torch.abs(pc_diff), dim=-1)  # (B, n, m)
    else:
        pc_dist = torch.sum(pc_diff ** 2, dim=-1)  # (B, n, m)
    dist1, idx1 = torch.min(pc_dist, dim=2)  # (B, n)
    dist2, idx2 = torch.min(pc_dist, dim=1)  # (B, m)
    return dist1, idx1, dist2, idx2


class GradReverse(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, eta=1.0):
        ctx.eta = eta
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output * -ctx.eta), None


def grad_reverse(x, eta=1.0):
    return GradReverse.apply(x, eta)


class VoteNetCriterion(nn.Module):

    def __init__(self,
                 num_classes,
                 num_heading_areas,
                 mean_sizes,
                 use_focal_loss=False,
                 gt_vote_factor=3,
                 near_threshold=0.3,
                 far_threshold=0.6,
                 coef_vote_loss=10.0,
                 coef_objectness_loss=5.0,
                 coef_center_loss=10.0,
                 coef_heading_cls_loss=1.0,
                 coef_heading_reg_loss=10.0,
                 coef_size_cls_loss=1.0,
                 coef_size_reg_loss=10.0 / 3.0,
                 coef_cls_loss=1.0,
                 device='cuda'):
        super().__init__()
        self.num_classes = num_classes
        self.num_heading_areas = num_heading_areas
        self.mean_sizes = mean_sizes
        self.use_focal_loss = use_focal_loss
        self.gt_vote_factor = gt_vote_factor
        self.near_threshold = near_threshold
        self.far_threshold = far_threshold
        # put larger weights on positive objectness
        self.obj_criterion = nn.CrossEntropyLoss(torch.tensor([0.2, 0.8], device=device), reduction='none')
        self.coef_vote_loss = coef_vote_loss
        self.coef_objectness_loss = coef_objectness_loss
        self.coef_center_loss = coef_center_loss
        self.coef_heading_cls_loss = coef_heading_cls_loss
        self.coef_heading_reg_loss = coef_heading_reg_loss
        self.coef_size_cls_loss = coef_size_cls_loss
        self.coef_size_reg_loss = coef_size_reg_loss
        self.coef_cls_loss = coef_cls_loss
        self.device = device

    def vote_loss(self, outputs, ground_truths):
        """
        Compute vote loss: Match predicted votes to GT votes.
        Overall idea:
                If the seed point belongs to an object (votes_label_mask == 1),
                then we require it to vote for the object center.
                Each seed point may vote for multiple translations v1,v2,v3
                A seed point may also be in the boxes of multiple objects:
                o1,o2,o3 with corresponding GT votes c1,c2,c3
                Then the loss for this seed point is:
                    min(d(v_i,c_j)) for i=1,2,3 and j=1,2,3
        :param outputs: dict of outputs
        :param ground_truths: dict of ground truths
        :return: vote_loss: vote loss
        """
        # Load ground truth votes and assign them to seed points
        batch_size, num_seed = outputs['seed_xyz'].shape[0], outputs['seed_xyz'].shape[1]
        vote_xyz, seed_inds = outputs['vote_xyz'], outputs['seed_inds'].long()
        # Get ground truth votes for the seed points
        # point_votes_mask: Use gather to select B,num_seed from B,num_point
        #   non-object point has no GT vote mask = 0, object point has mask = 1
        # point_votes: Use gather to select B,num_seed,9 from B,num_point,9
        #   with inds in shape B,num_seed,9 and 9 = GT_VOTE_FACTOR * 3
        seed_gt_votes_mask = torch.gather(ground_truths['point_votes_mask'], 1, seed_inds)
        seed_inds_expand = seed_inds.view(batch_size, num_seed, 1).repeat(1, 1, 3 * self.gt_vote_factor)
        seed_gt_votes = torch.gather(ground_truths['point_votes'], 1, seed_inds_expand)
        seed_gt_votes += outputs['seed_xyz'].repeat(1, 1, 3)
        # Compute the min of distance
        # from B,num_seed*vote_factor,3 to B*num_seed,vote_factor,3
        vote_xyz_reshape = vote_xyz.view(batch_size * num_seed, -1, 3)
        # from B,num_seed,3*GT_VOTE_FACTOR to B*num_seed,GT_VOTE_FACTOR,3
        seed_gt_votes_reshape = seed_gt_votes.view(batch_size * num_seed, self.gt_vote_factor, 3)
        # A predicted vote to nowhere is not penalized as long as there is a good vote near the GT vote.
        dist1, _, dist2, _ = nn_distance(vote_xyz_reshape, seed_gt_votes_reshape, l1=True)
        votes_dist, _ = torch.min(dist2, dim=1)  # (B*num_seed,vote_factor) to (B*num_seed,)
        votes_dist = votes_dist.view(batch_size, num_seed)
        vote_loss = torch.sum(votes_dist * seed_gt_votes_mask.float()) / (torch.sum(seed_gt_votes_mask.float()) + 1e-6)
        return vote_loss

    def objectness_loss(self, outputs, ground_truths):
        """
        Compute objectness loss for the proposals.
        :param outputs: dict of outputs
        :param ground_truths: dict of ground truths
        :return: objectness_loss: scalar Tensor
        :return: objectness_label: (batch_size, num_seed) Tensor with value 0 or 1
        :return: objectness_mask: (batch_size, num_seed) Tensor with value 0 or 1
        :return: object_assignment: (batch_size, num_seed) Tensor with long int within [0, num_gt_object - 1]
        """
        # Associate proposal and GT objects by point-to-point distances
        aggregated_vote_xyz = outputs['aggregated_vote_xyz']
        gt_center = ground_truths['centers'][:, :, 0: 3]
        b = gt_center.shape[0]
        k = aggregated_vote_xyz.shape[1]
        dist1, ind1, dist2, _ = nn_distance(aggregated_vote_xyz, gt_center)  # dist1: BxK, dist2: BxK2
        # Generate objectness label and mask
        # objectness_label: 1 if pred object center is within NEAR_THRESHOLD of any GT object
        # objectness_mask: 0 if pred object center is in gray zone (DO NOT CARE), 1 otherwise
        euclidean_dist1 = torch.sqrt(dist1 + 1e-6)
        objectness_label = torch.zeros((b, k), dtype=torch.long).to(self.device)
        objectness_mask = torch.zeros((b, k)).to(self.device)
        objectness_label[euclidean_dist1 < self.near_threshold] = 1
        objectness_mask[euclidean_dist1 < self.near_threshold] = 1
        objectness_mask[euclidean_dist1 > self.far_threshold] = 1
        # Compute objectness loss
        objectness_scores = outputs['objectness_scores']
        objectness_loss = self.obj_criterion(objectness_scores.transpose(2, 1), objectness_label)
        objectness_loss = torch.sum(objectness_loss * objectness_mask) / (torch.sum(objectness_mask) + 1e-6)
        # Set assignment
        object_assignment = ind1  # (b,k) with values in 0,1,...,K2-1
        return objectness_loss, objectness_label, objectness_mask, object_assignment

    @staticmethod
    def center_loss(outputs, ground_truths, objectness_label):
        """
        Compute center loss.
        :param outputs: dict of outputs
        :param ground_truths: dict of ground truths
        :param objectness_label: (batch_size, num_seed) Tensor with value 0 or 1
        :return: center_loss: center loss
        """
        pred_center, gt_center = outputs['centers'], ground_truths['centers'][:, :, 0: 3]
        dist1, ind1, dist2, _ = nn_distance(pred_center, gt_center)
        label_mask = ground_truths['label_masks']
        objectness_label = objectness_label.float()
        centroid_reg_loss1 = torch.sum(dist1 * objectness_label) / (torch.sum(objectness_label) + 1e-6)
        centroid_reg_loss2 = torch.sum(dist2 * label_mask) / (torch.sum(label_mask) + 1e-6)
        center_loss = centroid_reg_loss1 + centroid_reg_loss2
        return center_loss

    def heading_loss(self, outputs, ground_truths, objectness_label, object_assignment):
        """
        Compute heading loss.
        :param outputs: dict of outputs
        :param ground_truths: dict of ground truths
        :param objectness_label: (batch_size, num_seed) Tensor with value 0 or 1
        :param object_assignment: (batch_size, num_seed) Tensor with long int within [0, num_gt_object - 1]
        :return: heading_loss: heading loss
        """
        batch_size = object_assignment.shape[0]
        heading_area_gt = torch.gather(ground_truths['heading_area_ids'], 1, object_assignment)
        heading_cls_loss = cross_entropy(outputs['heading_scores'].transpose(2, 1), heading_area_gt, reduction='none')
        heading_cls_loss = torch.sum(heading_cls_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)
        heading_offsets_gt = torch.gather(ground_truths['heading_offsets'], 1, object_assignment)
        heading_offsets_gt_norm = heading_offsets_gt / (np.pi / self.num_heading_areas)
        heading_area_gt_hot = torch.zeros([batch_size, heading_area_gt.shape[1], self.num_heading_areas],
                                          dtype=torch.float32, device=self.device)
        heading_area_gt_hot.scatter_(2, heading_area_gt.unsqueeze(-1), 1)
        heading_reg_loss = huber_loss(
            torch.sum(outputs['heading_offsets_norm'] * heading_area_gt_hot, -1) - heading_offsets_gt_norm,
            delta=1.0
        )
        heading_reg_loss = torch.sum(heading_reg_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)
        return heading_cls_loss, heading_reg_loss

    def size_loss(self, outputs, ground_truths, objectness_label, object_assignment):
        """
        Compute size loss.
        :param outputs: dict of outputs
        :param ground_truths: dict of ground truths
        :param objectness_label: (batch_size, num_seed) Tensor with value 0 or 1
        :param object_assignment: (batch_size, num_seed) Tensor with long int within [0, num_gt_object - 1]
        :return: size_loss: size loss
        """
        batch_size = object_assignment.shape[0]
        class_label = torch.gather(ground_truths['categories'], 1, object_assignment)  # select (B,K) from (B,K2)
        size_cls_loss = cross_entropy(outputs['size_cls_scores'].transpose(2, 1), class_label, reduction='none')
        size_cls_loss = torch.sum(size_cls_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)
        size_offsets_gt = torch.gather(
            ground_truths['size_offsets'], 1, object_assignment.unsqueeze(-1).repeat(1, 1, 3))
        class_label_one_hot = torch.zeros([batch_size, class_label.shape[1], self.num_classes],
                                          dtype=torch.float32, device=self.device)
        class_label_one_hot.scatter_(2, class_label.unsqueeze(-1), 1)
        class_label_one_hot_tiled = class_label_one_hot.unsqueeze(-1).repeat(1, 1, 1, 3)
        size_offsets_norm = torch.sum(outputs['size_offsets_norm'] * class_label_one_hot_tiled, 2)
        mean_size = torch.from_numpy(self.mean_sizes.astype(np.float32)).to(self.device).unsqueeze(0).unsqueeze(0)
        mean_size = torch.sum(class_label_one_hot_tiled * mean_size, 2)  # (B,K,3)
        size_offsets_gt_norm = size_offsets_gt / mean_size  # (B,K,3)
        size_reg_loss = torch.mean(huber_loss(size_offsets_norm - size_offsets_gt_norm, delta=1.0), -1)
        size_reg_loss = torch.sum(size_reg_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)
        return size_cls_loss, size_reg_loss

    def box_loss(self, outputs, ground_truths, objectness_label, object_assignment):
        """
        Compute 3D bounding box loss.
        :param outputs: dict of outputs
        :param ground_truths: dict of ground truths
        :param objectness_label: (batch_size, num_seed) Tensor with value 0 or 1
        :param object_assignment: (batch_size, num_seed) Tensor with long int within [0, num_gt_object - 1]
        :return: box_loss: 3d bounding box loss
        """
        box_loss = self.center_loss(outputs, ground_truths, objectness_label) * self.coef_center_loss
        head_cls_loss, head_reg_loss = self.heading_loss(outputs, ground_truths, objectness_label, object_assignment)
        box_loss += head_cls_loss * self.coef_heading_cls_loss + head_reg_loss * self.coef_heading_reg_loss
        size_cls_loss, size_reg_loss = self.size_loss(outputs, ground_truths, objectness_label, object_assignment)
        box_loss += size_cls_loss * self.coef_size_cls_loss + size_reg_loss * self.coef_size_reg_loss
        return box_loss

    def semantic_cls_loss(self, outputs, ground_truths, objectness_label, object_assignment, gamma=2.0, alpha=0.25):
        """
        Compute semantic classification loss.
        :param outputs: dict of outputs
        :param ground_truths: dict of ground truths
        :param objectness_label: (batch_size, num_seed) Tensor with value 0 or 1
        :param object_assignment: (batch_size, num_seed) Tensor with long int within [0, num_gt_object - 1]
        :param gamma: gamma parameter in focal loss
        :param alpha: alpha parameter in focal loss
        :return: sem_cls_loss: semantic classification loss
        """
        sem_cls_label = torch.gather(ground_truths['categories'], 1, object_assignment)
        if self.use_focal_loss:
            sem_cls_probs = torch.sigmoid(outputs['sem_cls_scores'])
            # convert labels to one-hot labels
            sem_cls_label_one_hot = torch.zeros(sem_cls_probs.shape, device=self.device)
            sem_cls_label_one_hot.scatter_(2, sem_cls_label.unsqueeze(-1), 1)
            # compute focal loss
            ce_loss = binary_cross_entropy_with_logits(sem_cls_probs, sem_cls_label_one_hot, reduction="none")
            p_t = sem_cls_probs * sem_cls_label_one_hot + (1 - sem_cls_probs) * (1 - sem_cls_label_one_hot)
            sem_cls_loss = ce_loss * ((1 - p_t) ** gamma)
            if alpha >= 0:
                alpha_t = alpha * sem_cls_label_one_hot + (1 - alpha) * (1 - sem_cls_label_one_hot)
                sem_cls_loss = alpha_t * sem_cls_loss
            sem_cls_loss = torch.mean(sem_cls_loss, dim=-1)
        else:
            sem_cls_loss = cross_entropy(outputs['sem_cls_scores'].transpose(2, 1), sem_cls_label, reduction='none')
        objectness_label = objectness_label.float()
        sem_cls_loss = torch.sum(sem_cls_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)
        return sem_cls_loss

    @staticmethod
    def obj_acc(outputs, objectness_label, objectness_mask):
        """
        Compute objectness accuracy.
        :param outputs: dict of outputs
        :param objectness_label: (batch_size, num_seed) Tensor with value 0 or 1
        :param objectness_mask: (batch_size, num_seed) Tensor with value 0 or 1
        :return: obj_acc: objectness accuracy
        """
        objectness_label = objectness_label.long()
        obj_pred_val = torch.argmax(outputs['objectness_scores'], 2)
        obj_acc = torch.sum((obj_pred_val == objectness_label).float() * objectness_mask)
        obj_acc /= torch.sum(objectness_mask) + 1e-6
        return obj_acc

    def forward(self, outputs, ground_truths):
        """
        Loss functions of votenet
        :param outputs: dict of outputs
        :param ground_truths: dict of ground truths
        :return: loss: loss tensor for back propagation
        :return: loss_dict: dict of losses
        """
        # Vote loss
        if ground_truths['point_votes'] is not None:
            vote_loss = self.vote_loss(outputs, ground_truths)
        else:
            vote_loss = torch.tensor(0.0, device=self.device)
        # Objectness loss
        objectness_loss, objectness_label, objectness_mask, object_assignment = \
            self.objectness_loss(outputs, ground_truths)
        # num_proposals = float(objectness_label.shape[0] * objectness_label.shape[1])
        # pos_ratio = torch.sum(objectness_label.float().to(self.device)) / num_proposals
        # neg_ratio = torch.sum(objectness_mask.float()) / num_proposals - pos_ratio
        # Box loss
        box_loss = self.box_loss(outputs, ground_truths, objectness_label, object_assignment)
        # Semantic cls loss
        sem_cls_loss = self.semantic_cls_loss(outputs, ground_truths, objectness_label, object_assignment)
        # Final loss function
        loss = self.coef_vote_loss * vote_loss + self.coef_objectness_loss * objectness_loss
        loss += box_loss + self.coef_cls_loss * sem_cls_loss
        # --------------------------------------------
        # Some other statistics
        obj_acc = self.obj_acc(outputs, objectness_label, objectness_mask)
        loss_dict = {
            'vote_loss': vote_loss,
            'objectness_loss': objectness_loss,
            'box_loss': box_loss,
            'sem_cls_loss': sem_cls_loss,
            'loss': loss,
            'obj_acc': obj_acc,
        }
        return loss, loss_dict
