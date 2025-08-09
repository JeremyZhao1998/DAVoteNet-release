from typing import List, Tuple

import torch
import torch.nn as nn
from torch.nn.functional import relu, max_pool2d, avg_pool2d
from pytorch3d.ops import sample_farthest_points, knn_points, knn_gather, ball_query


class STNkd(nn.Module):
    def __init__(self, channel, k=64):
        super(STNkd, self).__init__()
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.k = k

    def forward(self, x):
        bs = x.size()[0]
        x = relu(self.bn1(self.conv1(x)))
        x = relu(self.bn2(self.conv2(x)))
        x = relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = relu(self.bn4(self.fc1(x)))
        x = relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        identity = torch.eye(self.k).view(1, self.k * self.k).repeat(bs, 1).to(x.device)
        x = x + identity
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, x_transform=False, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.x_transform = x_transform
        if self.x_transform:
            self.stn = STNkd(channel, k=3)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.feature_stn = STNkd(channel=64, k=64)

    def forward(self, x):
        if self.x_transform:
            _, dim_num, _ = x.size()
            trans = self.stn(x)
            x = x.transpose(2, 1)
            feature = x[:, :, 3:] if dim_num > 3 else None
            x = torch.bmm(x[:, :, :3], trans)
            if dim_num > 3:
                x = torch.cat([x, feature], dim=2)
            x = x.transpose(2, 1)
        x = relu(self.bn1(self.conv1(x)))
        if self.feature_transform:
            trans_feat = self.feature_stn(x)
            x = torch.bmm(x.transpose(2, 1), trans_feat).transpose(2, 1)
        x = relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        global_feature = torch.max(x, 2, keepdim=True)[0]
        return x, global_feature


class _BNBase(nn.Sequential):

    def __init__(self, in_size, batch_norm=None, name=""):
        super().__init__()
        self.add_module(name + "bn", batch_norm(in_size))
        nn.init.constant_(self[0].weight, 1.0)
        nn.init.constant_(self[0].bias, 0)


class BatchNorm2d(_BNBase):

    def __init__(self, in_size: int, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm2d, name=name)


class _ConvBase(nn.Sequential):

    def __init__(
            self,
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=None,
            batch_norm=None,
            bias=True,
            preact=False,
            name=""
    ):
        super().__init__()
        bias = bias and (not bn)
        conv_unit = conv(
            in_size,
            out_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        init(conv_unit.weight)
        if bias:
            nn.init.constant_(conv_unit.bias, 0)
        bn_unit = None
        if bn:
            if not preact:
                bn_unit = batch_norm(out_size)
            else:
                bn_unit = batch_norm(in_size)
        if preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)
            if activation is not None:
                self.add_module(name + 'activation', activation)
        self.add_module(name + 'conv', conv_unit)
        if not preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)
            if activation is not None:
                self.add_module(name + 'activation', activation)


class Conv2d(_ConvBase):

    def __init__(
            self,
            in_size: int,
            out_size: int,
            *,
            kernel_size: Tuple[int, int] = (1, 1),
            stride: Tuple[int, int] = (1, 1),
            padding: Tuple[int, int] = (0, 0),
            activation=nn.ReLU(inplace=True),
            bn: bool = False,
            init=nn.init.kaiming_normal_,
            bias: bool = True,
            preact: bool = False,
            name: str = ""
    ):
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv2d,
            batch_norm=BatchNorm2d,
            bias=bias,
            preact=preact,
            name=name
        )


class SharedMLP(nn.Sequential):

    def __init__(
            self,
            args: List[int],
            *,
            bn: bool = False,
            activation=nn.ReLU(inplace=True),
            preact: bool = False,
            first: bool = False,
            name: str = ""
    ):
        super().__init__()
        for i in range(len(args) - 1):
            self.add_module(
                name + 'layer{}'.format(i),
                Conv2d(
                    args[i],
                    args[i + 1],
                    bn=(not first or not preact or (i != 0)) and bn,
                    activation=activation
                    if (not first or not preact or (i != 0)) else None,
                    preact=preact
                )
            )


class QueryAndGroup(nn.Module):
    """
    Groups with a ball query of radius

    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(
            self,
            radius,
            nsample,
            use_xyz=True,
            ret_grouped_xyz=False,
            normalize_xyz=False,
            sample_uniformly=False,
            ret_unique_cnt=False
    ):
        super(QueryAndGroup, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz
        self.normalize_xyz = normalize_xyz
        self.sample_uniformly = sample_uniformly
        self.ret_unique_cnt = ret_unique_cnt
        if self.ret_unique_cnt:
            assert self.sample_uniformly

    def forward(self, xyz, new_xyz, features=None):
        """
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """
        idx_new = ball_query(new_xyz, xyz, K=self.nsample, radius=self.radius).idx
        # Deal with padding: change -1 padding into the first index
        idx_mask = (idx_new == torch.ones_like(idx_new) * -1)
        idx_pad = idx_new[:, :, 0].unsqueeze(2).repeat(1, 1, self.nsample)
        idx = idx_new * (~idx_mask) + idx_pad * idx_mask
        unique_cnt = None
        if self.sample_uniformly:
            unique_cnt = torch.zeros((idx.shape[0], idx.shape[1]))
            for i_batch in range(idx.shape[0]):
                for i_region in range(idx.shape[1]):
                    unique_ind = torch.unique(idx[i_batch, i_region, :])
                    num_unique = unique_ind.shape[0]
                    unique_cnt[i_batch, i_region] = num_unique
                    sample_ind = torch.randint(0, num_unique, (self.nsample - num_unique,), dtype=torch.long)
                    all_ind = torch.cat((unique_ind, unique_ind[sample_ind]))
                    idx[i_batch, i_region, :] = all_ind
        # xyz_trans = xyz.transpose(1, 2).contiguous()
        # grouped_xyz = grouping_operation(xyz_trans, idx.to(torch.int32))  # (B, 3, npoint, nsample)
        grouped_xyz = knn_gather(xyz, idx).permute(0, 3, 1, 2)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)
        if self.normalize_xyz:
            grouped_xyz /= self.radius
        if features is not None:
            grouped_features = knn_gather(features.transpose(1, 2), idx).permute(0, 3, 1, 2)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz
        ret = [new_features]
        if self.ret_grouped_xyz:
            ret.append(grouped_xyz)
        if self.ret_unique_cnt:
            ret.append(unique_cnt)
        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)


class GroupAll(nn.Module):
    """
    Groups all features
    """

    def __init__(self, use_xyz=True):
        super(GroupAll, self).__init__()
        self.use_xyz = use_xyz

    def forward(self, xyz, new_xyz, features=None):
        """
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            Ignored
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, C + 3, 1, N) tensor
        """

        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz
        return new_features


class PointnetSAModuleVotes(nn.Module):
    """
    Modified based on _PointnetSAModuleBase and PointnetSAModuleMSG
    with extra support for returning point indices for getting their GT votes
    """

    def __init__(
            self,
            mlp: List[int],
            npoint: int = None,
            radius: float = None,
            nsample: int = None,
            bn: bool = True,
            use_xyz: bool = True,
            pooling: str = 'max',
            sigma: float = None,  # for RBF pooling
            normalize_xyz: bool = False,  # noramlize local XYZ with radius
            sample_uniformly: bool = False,
            ret_unique_cnt: bool = False
    ):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.pooling = pooling
        self.mlp_module = None
        self.use_xyz = use_xyz
        self.sigma = sigma
        if self.sigma is None:
            self.sigma = self.radius / 2
        self.normalize_xyz = normalize_xyz
        self.ret_unique_cnt = ret_unique_cnt
        if npoint is not None:
            self.grouper = QueryAndGroup(radius, nsample, use_xyz=use_xyz, ret_grouped_xyz=True,
                                         normalize_xyz=normalize_xyz, sample_uniformly=sample_uniformly,
                                         ret_unique_cnt=ret_unique_cnt)
        else:
            self.grouper = GroupAll(use_xyz)
        mlp_spec = mlp
        if use_xyz and len(mlp_spec) > 0:
            mlp_spec[0] += 3
        self.mlp_module = SharedMLP(mlp_spec, bn=bn)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None, inds: torch.Tensor = None):
        """
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates
        features : torch.Tensor
            (B, C, N) tensor of the descriptors of the features
        inds : torch.Tensor
            (B, npoint) tensor that stores index to the xyz points (values in 0-N-1)

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B, sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        inds: torch.Tensor
            (B, npoint) tensor of the inds
        """
        if inds is None:
            _, inds = sample_farthest_points(xyz, K=self.npoint)
        else:
            assert (inds.shape[1] == self.npoint)
        new_xyz = torch.gather(xyz, dim=1, index=inds.unsqueeze(-1).repeat(1, 1, xyz.shape[-1]))
        if not self.ret_unique_cnt:
            grouped_features, grouped_xyz = self.grouper(xyz, new_xyz, features)  # (B, C, npoint, nsample)
            unique_cnt = None
        else:
            # (B, C, npoint, nsample), (B, 3, npoint,nsample), (B, npoint)
            grouped_features, grouped_xyz, unique_cnt = self.grouper(xyz, new_xyz, features)
        new_features = self.mlp_module(grouped_features)  # (B, mlp[-1], npoint, nsample)
        if self.pooling == 'max':
            new_features = max_pool2d(new_features, kernel_size=[1, new_features.size(3)])  # (B, mlp[-1], npoint, 1)
        elif self.pooling == 'avg':
            new_features = avg_pool2d(new_features, kernel_size=[1, new_features.size(3)])  # (B, mlp[-1], npoint, 1)
        elif self.pooling == 'rbf':
            # Use radial basis function kernel for weighted sum of features (normalized by nsample and sigma)
            # Ref: https://en.wikipedia.org/wiki/Radial_basis_function_kernel
            # (B, npoint, nsample)
            rbf = torch.exp(-1 * grouped_xyz.pow(2).sum(1, keepdim=False) / (self.sigma ** 2) / 2)
            # (B, mlp[-1], npoint, 1)
            new_features = torch.sum(new_features * rbf.unsqueeze(1), -1, keepdim=True) / float(self.nsample)
        new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
        if not self.ret_unique_cnt:
            return new_xyz, new_features, inds
        else:
            return new_xyz, new_features, inds, unique_cnt


class PointnetFPModule(nn.Module):
    r"""Propigates the features of one set to another

    Parameters
    ----------
    mlp :
        Pointnet module parameters
    bn : bool
        Use batchnorm
    """

    def __init__(self, *, mlp: List[int], bn: bool = True):
        super().__init__()
        self.mlp = SharedMLP(mlp, bn=bn)

    def forward(self, unknown: torch.Tensor, known: torch.Tensor,
                unknow_feats: torch.Tensor, known_feats: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of the xyz positions of the unknown features
        known : torch.Tensor
            (B, m, 3) tensor of the xyz positions of the known features
        unknow_feats : torch.Tensor
            (B, C1, n) tensor of the features to be propagated to
        known_feats : torch.Tensor
            (B, C2, m) tensor of features to be propagated

        Returns
        -------
        new_features : torch.Tensor
            (B, mlp[-1], n) tensor of the features of the unknown features
        """
        if known is not None:
            # dist, idx = functions.three_nn(unknown, known)
            knn = knn_points(unknown, known, K=3)
            dist, idx = torch.sqrt(knn.dists), knn.idx
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            # interpolated_feats = functions.three_interpolate(known_feats, idx.to(torch.int32), weight)
            feats = knn_gather(known_feats.transpose(1, 2), idx)
            interpolated_feats = torch.sum(feats * weight.unsqueeze(-1), dim=2, keepdim=False).transpose(1, 2)
        else:
            interpolated_feats = known_feats.expand(*known_feats.size()[0:2], unknown.size(1))
        if unknow_feats is not None:
            new_features = torch.cat([interpolated_feats, unknow_feats], dim=1)  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats
        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)
        return new_features.squeeze(-1)


class Pointnet2Backbone(nn.Module):
    """
    Backbone network for point cloud feature learning.
    Based on Pointnet++ single-scale grouping network.

    Parameters
    ----------
    input_feature_dim: int
        Number of input channels in the feature descriptor for each point. e.g. 3 for RGB.
    """

    def __init__(self, input_feature_dim=0):
        super().__init__()
        self.sa1 = PointnetSAModuleVotes(
            npoint=2048,
            radius=0.2,
            nsample=64,
            mlp=[input_feature_dim, 64, 64, 128],
            use_xyz=True,
            normalize_xyz=True
        )
        self.sa2 = PointnetSAModuleVotes(
            npoint=1024,
            radius=0.4,
            nsample=32,
            mlp=[128, 128, 128, 256],
            use_xyz=True,
            normalize_xyz=True
        )
        self.sa3 = PointnetSAModuleVotes(
            npoint=512,
            radius=0.8,
            nsample=16,
            mlp=[256, 128, 128, 256],
            use_xyz=True,
            normalize_xyz=True
        )
        self.sa4 = PointnetSAModuleVotes(
            npoint=256,
            radius=1.2,
            nsample=16,
            mlp=[256, 128, 128, 256],
            use_xyz=True,
            normalize_xyz=True
        )
        self.fp1 = PointnetFPModule(mlp=[256 + 256, 256, 256])
        self.fp2 = PointnetFPModule(mlp=[256 + 256, 256, 256])

    @staticmethod
    def _break_up_pc(pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )
        return xyz, features

    def forward(self, point_cloud, end_points=None):
        """
        Forward pass of the network

        Parameters
        ----------
        point_cloud: Variable(torch.Tensor)
            (B, N, 3 + input_feature_dim) tensor
            Point cloud to run predicts on
            Each point in the point-cloud MUST
            be formated as (x, y, z, features...)
        end_points: dict

        Returns
        ----------
        end_points: {XXX_xyz, XXX_features, XXX_inds}
            XXX_xyz: float32 Tensor of shape (B,K,3)
            XXX_features: float32 Tensor of shape (B,K,D)
            XXX-inds: int64 Tensor of shape (B,K) values in [0,N-1]
        """
        if not end_points:
            end_points = {}
        xyz, features = self._break_up_pc(point_cloud)
        # --------- 4 SET ABSTRACTION LAYERS ---------
        xyz, features, fps_inds = self.sa1(xyz, features)
        end_points['sa1_inds'] = fps_inds
        end_points['sa1_xyz'] = xyz
        end_points['sa1_features'] = features
        xyz, features, fps_inds = self.sa2(xyz, features)  # this fps_inds is just 0,1,...,1023
        end_points['sa2_inds'] = fps_inds
        end_points['sa2_xyz'] = xyz
        end_points['sa2_features'] = features
        xyz, features, fps_inds = self.sa3(xyz, features)  # this fps_inds is just 0,1,...,511
        end_points['sa3_xyz'] = xyz
        end_points['sa3_features'] = features
        xyz, features, fps_inds = self.sa4(xyz, features)  # this fps_inds is just 0,1,...,255
        end_points['sa4_xyz'] = xyz
        end_points['sa4_features'] = features
        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        features = self.fp1(end_points['sa3_xyz'], end_points['sa4_xyz'], end_points['sa3_features'],
                            end_points['sa4_features'])
        features = self.fp2(end_points['sa2_xyz'], end_points['sa3_xyz'], end_points['sa2_features'], features)
        end_points['fp2_features'] = features
        end_points['fp2_xyz'] = end_points['sa2_xyz']
        num_seed = end_points['fp2_xyz'].shape[1]
        end_points['fp2_inds'] = end_points['sa1_inds'][:, 0:num_seed]  # indices among the entire input point clouds
        return end_points
