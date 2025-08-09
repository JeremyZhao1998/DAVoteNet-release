import torch
import torch.nn as nn
from torch.nn.functional import relu

from models.pointnet2 import Pointnet2Backbone, PointnetSAModuleVotes
from pytorch3d.ops import sample_farthest_points
import utils.bbox_utils as bbox_utils


class VotingModule(nn.Module):
    """
    Votes generation from seed point features.

    Arguments
    ----------
    vote_factor: int
        number of votes generated from each seed point
    seed_feature_dim: int
        number of channels of seed point features
    """

    def __init__(self, vote_factor, seed_feature_dim):
        super().__init__()
        self.vote_factor = vote_factor
        self.in_dim = seed_feature_dim
        self.out_dim = self.in_dim  # due to residual feature, in_dim has to be == out_dim
        self.conv1 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv3 = torch.nn.Conv1d(self.in_dim, (3 + self.out_dim) * self.vote_factor, 1)
        self.bn1 = torch.nn.BatchNorm1d(self.in_dim)
        self.bn2 = torch.nn.BatchNorm1d(self.in_dim)

    def forward(self, seed_xyz, seed_features):
        """
        Forward pass.

        Arguments:
            seed_xyz: (batch_size, num_seed, 3) Pytorch tensor
            seed_features: (batch_size, feature_dim, num_seed) Pytorch tensor
        Returns:
            vote_xyz: (batch_size, num_seed*vote_factor, 3)
            vote_features: (batch_size, vote_feature_dim, num_seed*vote_factor)
        """
        batch_size = seed_xyz.shape[0]
        num_seed = seed_xyz.shape[1]
        num_vote = num_seed * self.vote_factor
        net = relu(self.bn1(self.conv1(seed_features)))
        net = relu(self.bn2(self.conv2(net)))
        net = self.conv3(net)  # (batch_size, (3+out_dim)*vote_factor, num_seed)
        net = net.transpose(2, 1).view(batch_size, num_seed, self.vote_factor, 3 + self.out_dim)
        offset = net[:, :, :, 0:3]
        vote_xyz = seed_xyz.unsqueeze(2) + offset
        vote_xyz = vote_xyz.contiguous().view(batch_size, num_vote, 3)
        residual_features = net[:, :, :, 3:]  # (batch_size, num_seed, vote_factor, out_dim)
        vote_features = seed_features.transpose(2, 1).unsqueeze(2) + residual_features
        vote_features = vote_features.contiguous().view(batch_size, num_vote, self.out_dim)
        vote_features = vote_features.transpose(2, 1).contiguous()
        return vote_xyz, vote_features


class ProposalModule(nn.Module):
    """
    Object proposal module.

    Arguments
    ----------
    num_classes: int
        Number of semantics classes to predict over -- size of softmax classifier
    num_heading_areas: int
        Number of areas to use for heading prediction
    mean_sizes: np.ndarray
        Mean size of objects in the dataset.  Should be a num_classes x 3 array
    num_proposals: int
        Number of proposals/detections generated from the network. Each proposal is a 3D OBB with a semantic class.
    sampling: str
        Sampling strategy for generating proposals.  Must be one of ['vote_fps', 'seed_fps', 'scannet8']
    """

    def __init__(self,
                 num_classes,
                 num_heading_areas,
                 mean_sizes,
                 num_proposals,
                 sampling,
                 seed_feat_dim=256):
        super().__init__()
        self.num_classes = num_classes
        self.num_heading_areas = num_heading_areas
        self.mean_sizes = mean_sizes
        self.num_proposals = num_proposals
        self.sampling = sampling
        self.seed_feat_dim = seed_feat_dim
        # Vote clustering
        self.vote_aggregation = PointnetSAModuleVotes(
            npoint=self.num_proposals,
            radius=0.3,
            nsample=16,
            mlp=[self.seed_feat_dim, 128, 128, 128],
            use_xyz=True,
            normalize_xyz=True
        )
        self.conv1 = torch.nn.Conv1d(128, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 2 + 3 + num_heading_areas * 2 + num_classes * 4 + num_classes, 1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(128)

    def decode_raw_proposals(self, raw_proposals, aggregated_vote_xyz):
        bs, np = raw_proposals.shape[0], raw_proposals.shape[1]
        nh, nc = self.num_heading_areas, self.num_classes
        outputs = {
            'objectness_scores': raw_proposals[:, :, 0: 2],
            'centers': aggregated_vote_xyz + raw_proposals[:, :, 2: 5],
            'heading_scores': raw_proposals[:, :, 5: 5 + nh],
            'heading_offsets_norm': raw_proposals[:, :, 5 + nh: 5 + nh * 2],
            'size_cls_scores': raw_proposals[:, :, 5 + nh * 2: 5 + nh * 2 + nc],
            'size_offsets_norm': raw_proposals[:, :, 5 + nh * 2 + nc: 5 + nh * 2 + nc * 4].view([bs, np, nc, 3]),
            'sem_cls_scores': raw_proposals[:, :, 5 + nh * 2 + nc * 4:]
        }
        outputs['heading_offsets'] = outputs['heading_offsets_norm'] * (torch.pi / nh)
        mean_sizes = self.mean_sizes.to(raw_proposals.device)
        outputs['size_offsets'] = outputs['size_offsets_norm'] * mean_sizes.unsqueeze(0).unsqueeze(0)
        return outputs

    def forward(self, xyz, features, seed_xyz):
        """
        Args:
            xyz: (B, K, 3)
            features: (B, C, K)
            seed_xyz: (B, num_seed, 3)
        Returns:
            xyz: aggregated_vote_xyz: (B, num_proposal, 3)
            sample_inds: aggregated_vote_inds: (B, num_proposal)
            raw_proposals: (B, num_proposal, 2 + 3 + num_heading_areas * 2 + num_classes * 4 + num_classes)
                2 for objectness scores,
                3 for centers,
                num_heading_areas * 2 for heading area classification scores and heading offsets regressions,
                num_classes * 4 for size classification scores and size offsets regressions,
                num_classes for semantic classification scores
        """
        if self.sampling == 'vote_fps':
            # Farthest point sampling (FPS) on votes
            xyz, features, fps_inds = self.vote_aggregation(xyz, features)
            sample_inds = fps_inds
        elif self.sampling == 'seed_fps':
            # FPS on seed and choose the votes corresponding to the seeds
            # This gets us a slightly better coverage of *object* votes than vote_fps
            # (which tends to get more cluster votes)
            # sample_inds = furthest_point_sample(seed_xyz, self.num_proposals)
            _, sample_inds = sample_farthest_points(seed_xyz, K=self.num_proposals)
            xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)
        elif self.sampling == 'random':
            # Random sampling from the votes
            batch_size, num_seed = seed_xyz.shape[0], seed_xyz.shape[1]
            sample_inds = torch.randint(0, num_seed, (batch_size, self.num_proposals), dtype=torch.int).cuda()
            xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)
        else:
            print('Unknown sampling strategy: %s. Exiting!' % self.sampling)
            exit()
        # --------- PROPOSAL GENERATION ---------
        net = relu(self.bn1(self.conv1(features)))
        net = relu(self.bn2(self.conv2(net)))
        raw_proposals = self.conv3(net)
        return xyz, sample_inds, raw_proposals.transpose(2, 1)


class VoteNet(nn.Module):
    """
    A deep neural network for 3D object detection with end-to-end optimizable hough voting.
    """

    def __init__(self,
                 num_classes,
                 num_heading_areas,
                 mean_sizes,
                 backbone='pointnet2',
                 axis_aligned=False,
                 input_feature_dim=0,
                 num_proposals=128,
                 vote_factor=1,
                 gt_vote_factor=3,
                 sampling='vote_fps'):
        super().__init__()
        assert (mean_sizes.shape[0] == num_classes)
        self.num_heading_areas = num_heading_areas
        self.mean_sizes = torch.tensor(mean_sizes, dtype=torch.float32)
        self.axis_aligned = axis_aligned
        # Backbone point feature learning
        if backbone == 'pointnet2':
            self.backbone_net = Pointnet2Backbone(input_feature_dim)
        else:
            raise NotImplementedError('Backbone not supported: %s' % backbone)
        # Hough voting
        self.voting_module = VotingModule(vote_factor, 256)
        # Vote aggregation and detection
        self.proposal_module = ProposalModule(num_classes, num_heading_areas, self.mean_sizes, num_proposals, sampling)
        self.gt_vote_factor = gt_vote_factor

    @torch.no_grad()
    def post_process(self, point_clouds, outputs, obj_threshold=0.05,
                     sem_threshold=0.0, nms_threshold=0.25, get_votes=False):
        centers = outputs['centers']
        batch_size, num_proposals = centers.shape[0], centers.shape[1]
        heading_cls = torch.argmax(outputs['heading_scores'], -1)
        heading_offsets = torch.gather(outputs['heading_offsets'], 2, heading_cls.unsqueeze(-1)).squeeze_(2)
        heading_angles = bbox_utils.encode_angles(heading_cls, heading_offsets, self.num_heading_areas)
        size_cls = torch.argmax(outputs['size_cls_scores'], -1)
        size_offsets = torch.gather(
            outputs['size_offsets'], 2, size_cls.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 3)).squeeze_(2)
        mean_sizes = self.mean_sizes.to(size_cls.device)[size_cls.view(-1), :].view(batch_size, num_proposals, 3)
        box_sizes = mean_sizes + size_offsets
        corners = bbox_utils.boxes_to_corners_3d(centers, heading_angles, box_sizes)
        # Remove empty box
        label_masks, point_in = bbox_utils.remove_empty_box(corners, point_clouds, axis_aligned=self.axis_aligned)
        # Remove boxes with too little size
        l, w, h = box_sizes[:, :, 0], box_sizes[:, :, 1], box_sizes[:, :, 2]
        label_masks = label_masks * (l >= 1e-2) * (w >= 1e-2) * (h >= 1e-2)
        # Remove boxes with low scores
        obj_probs = torch.softmax(outputs['objectness_scores'], -1)[:, :, 1]
        sem_cls_probs = torch.softmax(outputs['sem_cls_scores'], -1)
        label_masks = label_masks * (obj_probs > obj_threshold) * (sem_cls_probs.max(-1)[0] > sem_threshold)
        # Remove category 'others' boxes
        categories = torch.argmax(outputs['sem_cls_scores'], -1)
        label_masks = label_masks * (categories != self.proposal_module.num_classes - 1)
        # None maximum suppression
        label_masks = bbox_utils.nms_3d(corners, obj_probs, categories, label_masks, nms_threshold)
        # Generate votes
        point_votes, point_votes_mask = None, None
        if get_votes:
            point_votes = centers.unsqueeze(2) - point_clouds[:, :, : 3].unsqueeze(1)
            point_votes_mask = label_masks.unsqueeze(-1) * point_in
            box_choice = torch.argmax(point_votes_mask.float() * obj_probs.unsqueeze(-1), dim=1)
            tmp_mask = torch.zeros_like(point_votes_mask)
            for b in range(batch_size):
                tmp_matrix = tmp_mask[b]
                tmp_matrix[box_choice[b], torch.arange(point_clouds.shape[1])] = 1
                tmp_mask[b] = tmp_matrix
            point_votes_mask = point_votes_mask * tmp_mask
            point_votes = torch.sum(point_votes * point_votes_mask.unsqueeze(-1).float(), dim=1)
            point_votes = torch.concatenate([point_votes for _ in range(self.gt_vote_factor)], dim=-1)
            point_votes_mask = torch.sum(point_votes_mask, dim=1)
        # Generate final predictions
        predictions = {
            'point_clouds': point_clouds,
            'categories': categories,
            'sem_cls_probs': sem_cls_probs,
            'obj_probs': obj_probs,
            'centers': centers,
            'heading_angles': heading_angles,
            'heading_area_ids': heading_cls,
            'heading_offsets': heading_offsets,
            'box_sizes': box_sizes,
            'size_offsets': size_offsets,
            'corners': corners,
            'label_masks': label_masks,
            'point_votes': point_votes,
            'point_votes_mask': point_votes_mask
        }
        return predictions

    def forward(self, point_clouds, detection_outputs=True):
        """
        Forward pass of the network

        Args:
            point_clouds: torch.Tensor
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formatted as (x, y, z, features...)
            detection_outputs: bool
                Whether to return the detection outputs
        Returns:
            outputs: dict
        """
        # Backbone forward
        back_out = self.backbone_net(point_clouds)
        seed_xyz, seed_features, seed_inds = back_out['fp2_xyz'], back_out['fp2_features'], back_out['fp2_inds']
        outputs = {
            'seed_xyz': seed_xyz,
            'seed_inds': seed_inds,
            'seed_features': seed_features
        }
        if not detection_outputs:
            return outputs
        # Voting module forward
        vote_xyz, vote_features = self.voting_module(seed_xyz, seed_features)
        vote_features_norm = torch.norm(vote_features, p=2, dim=1)
        vote_features = vote_features.div(vote_features_norm.unsqueeze(1))
        outputs['vote_xyz'] = vote_xyz
        # Proposal module forward
        aggregated_vote_xyz, aggregated_vote_inds, raw_proposals = self.proposal_module(
            vote_xyz, vote_features, seed_xyz)
        outputs['aggregated_vote_xyz'] = aggregated_vote_xyz
        outputs.update(self.proposal_module.decode_raw_proposals(raw_proposals, aggregated_vote_xyz))
        return outputs
