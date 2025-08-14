import os
import random
import numpy as np
from torch.utils.data import Dataset

import utils.pc_utils as pc_utils
import utils.bbox_utils as bbox_utils
import pc_datasets.augmentations as aug


class PCDetectionDataset(Dataset):

    def __init__(self,
                 data_root,
                 dataset_name,
                 categories,
                 split_set='train',
                 num_points=20000,
                 max_num_obj=64,
                 axis_aligned=False,
                 num_heading_areas=12,
                 use_color=False,
                 use_height=False,
                 augment=False,
                 few_shot=-1,
                 return_votes=False,
                 vote_factor=3):
        # Get scan names
        self.dataset_name = dataset_name
        data_folder = 'pc_bboxes_' + ('axis_aligned_' if axis_aligned else '')
        self.data_path = os.path.join(data_root, dataset_name, data_folder + split_set)
        self.scan_names = sorted(list(set([os.path.basename(x)[: -14]
                                           for x in os.listdir(str(self.data_path)) if x != 'info'])))
        if few_shot > 0:
            self.scan_names = random.sample(self.scan_names, few_shot)
        # Category dictionary
        self.category_dict = {name: idx for idx, name in enumerate(categories)}
        # Point cloud basic information
        self.num_points = num_points
        self.max_num_obj = max_num_obj
        self.info_path = os.path.join(data_root, dataset_name, data_folder + 'train', 'info')
        self.mean_color_rgb = np.load(os.path.join(str(self.info_path), 'mean_color.npz'))['mean_color']
        self.augment = augment
        self.use_color = use_color
        self.use_height = use_height
        # Category information
        self.mean_sizes = np.zeros((len(self.category_dict), 3))
        mean_sizes = np.load(os.path.join(str(self.info_path), 'mean_sizes.npz'))
        for name, idx in self.category_dict.items():
            self.mean_sizes[idx, :] = mean_sizes[name] if name in mean_sizes else np.zeros((3,))
        self.mean_sizes[-1] = np.mean(self.mean_sizes[:-1], axis=0)
        self.num_classes = len(self.category_dict)
        # Bounding box information
        self.axis_aligned = axis_aligned
        self.num_heading_areas = num_heading_areas
        self.heading_area_size = 2 * np.pi / float(self.num_heading_areas)
        self.return_votes = return_votes
        self.vote_factor = vote_factor

    def __len__(self):
        return len(self.scan_names)

    def prepare_data_for_vis(self, idx, obj_only=False):
        sample = self[idx]
        pc = sample['point_clouds'][:, : 6]
        pc[:, 3:] = np.clip(pc[:, 3:] + self.mean_color_rgb, 0.0, 1.0)
        label_mask = sample['label_masks']
        corners = sample['corners'][label_mask == 1, :, :]
        if obj_only:
            pc_in_box = np.zeros(pc.shape[0], dtype=bool)
            for box in corners:
                pc_in_box = np.bitwise_or(pc_in_box, pc_utils.extract_pc_in_box3d(pc, box)[1])
            pc = pc[pc_in_box, :]
        return pc, corners

    def augment_data(self, point_cloud, bboxes):
        point_cloud, bboxes = aug.random_flip_yz(point_cloud, bboxes)
        point_cloud, bboxes = aug.random_flip_xz(point_cloud, bboxes)
        point_cloud, bboxes = aug.random_scale(point_cloud, bboxes, self.use_height)
        if not self.axis_aligned:
            point_cloud, bboxes = aug.random_rotate(point_cloud, bboxes, angle_range=np.pi / 3)
        if self.use_color:
            point_cloud = aug.color_augmentation(point_cloud, self.mean_color_rgb)
        return point_cloud, bboxes

    def _clean_bboxes(self, bboxes, category_names):
        kept_box_ids, categories = [], []
        for idx, name in enumerate(category_names):
            if name in self.category_dict:
                kept_box_ids.append(idx)
                categories.append(self.category_dict[name])
            elif self.augment and random.random() < 0.05:
                kept_box_ids.append(idx)
                categories.append(self.category_dict['others'])
        if len(kept_box_ids) == 0:
            for idx, name in enumerate(category_names):
                if name not in self.category_dict:
                    kept_box_ids.append(idx)
                    categories.append(self.category_dict['others'])
                    break
        if len(kept_box_ids) > self.max_num_obj:
            kept_box_ids = kept_box_ids[: self.max_num_obj]
            categories = categories[: self.max_num_obj]
        categories = np.array(categories)
        bboxes = bboxes[kept_box_ids, :]
        assert len(bboxes) == len(categories)
        return bboxes, categories

    def get_point_votes(self, point_cloud, centers, corners, categories):
        point_votes, point_votes_mask = np.zeros((self.num_points, self.vote_factor * 3)), np.zeros((self.num_points,))
        label_mask = np.full((self.num_points,), -1, dtype=np.int64)
        for center, corner, category in zip(centers, corners, categories):
            vote_xyz = np.expand_dims(center, 0) - point_cloud[:, : 3]
            try:
                _, inds = pc_utils.extract_pc_in_box3d_aligned(point_cloud, corner) if self.axis_aligned else \
                    pc_utils.extract_pc_in_box3d(point_cloud, corner)
            except ValueError:
                continue
            for v in range(self.vote_factor):
                v_inds = np.bitwise_and(inds, point_votes_mask == v)
                point_votes[v_inds, v * 3: (v + 1) * 3] = vote_xyz[v_inds, :]
            point_votes_mask[inds] += 1
            label_mask[inds] = category
        for v in range(1, self.vote_factor):
            vote_inds = point_votes_mask == v
            for vj in range(v, self.vote_factor):
                point_votes[vote_inds, vj * 3: (vj + 1) * 3] = point_votes[vote_inds, (v - 1) * 3: v * 3]
        point_votes_mask = np.array(point_votes_mask > 0, dtype=np.int64)
        return point_votes, point_votes_mask, label_mask

    def __getitem__(self, idx):
        """
        Returns a dict with following keys:
            point_clouds: (N, 3 + C)
            object_categories: (MAX_NUM_OBJ,)
            box_centers: (MAX_NUM_OBJ, 3) for GT box center XYZ
            heading_area_ids: (MAX_NUM_OBJ,) with int values in 0, ..., NUM_HEADING_BIN - 1
            heading_offsets: (MAX_NUM_OBJ,)
            box_sizes: (MAX_NUM_OBJ, 3) for GT box size l,w,h (minus by mean values)
            label_mask: (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique box
            original_bboxes: (MAX_NUM_OBJ, 8) for GT box corners XYZ, l,w,h, heading_angle, category
            point_votes: (N,9) with votes XYZ (3 votes: X1Y1Z1, X2Y2Z2, X3Y3Z3)
                if there is only one vote than X1==X2==X3 etc.
            point_votes_mask: (N, ) with 0/1 with 1 indicating the point
                is in one of the object's OBB.
            scan_idx: int scan index in scan_names list
            scan_name: str scan name
        """
        scan_name = self.scan_names[idx]
        data_dict = np.load(os.path.join(str(self.data_path), scan_name) + '_pc_bboxes.npz')
        point_cloud, bboxes, category_names = data_dict['pc'], data_dict['bboxes'], data_dict['categories']
        bboxes, categories = self._clean_bboxes(bboxes, category_names)
        if self.use_color:
            point_cloud = point_cloud[:, 0: 6]
            point_cloud[:, 3:6] = point_cloud[:, 3:6] - self.mean_color_rgb
        else:
            point_cloud = point_cloud[:, 0: 3]
        if self.use_height:
            floor_height = np.percentile(point_cloud[:, 2], 0.99)
            height = point_cloud[:, 2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1)  # (N, 4) or (N, 7)
        # # Data augmentations
        if self.augment:
            point_cloud, bboxes = self.augment_data(point_cloud, bboxes)
        if point_cloud.shape[0] != self.num_points:
            point_cloud, choices = pc_utils.random_sampling(point_cloud, self.num_points,
                                                            seed=sum([ord(c) for c in scan_name]), return_choices=True)
        # Labels
        # # Box centers
        centers = bboxes[:, : 3]
        # # Heading direction (decoded into heading_area_id and heading_offset)
        # # heading_angle = heading_area_ids * heading_area_size + heading_offset
        heading_angles = bboxes[:, 6] if not self.axis_aligned else np.zeros((bboxes.shape[0],))
        heading_area_ids, heading_offsets = bbox_utils.decode_angles(heading_angles, self.num_heading_areas)
        # # Box size (offset to the mean size of each category)
        box_sizes = bboxes[:, 3: 6]
        size_offsets = box_sizes - self.mean_sizes[[int(category) for category in categories], :]
        corners = np.stack([bbox_utils.boxes_to_corners_3d_np(c, a, s)
                            for c, a, s in zip(centers, heading_angles, box_sizes)], axis=0) \
            if len(bboxes) > 0 else np.zeros((0, 8, 3))
        # # Votes
        if self.return_votes:
            point_votes, point_votes_mask, label_mask = self.get_point_votes(point_cloud, centers, corners, categories)
        else:
            point_votes, point_votes_mask, label_mask = None, None, None
        # # Padding
        box_num, padding_num = bboxes.shape[0], self.max_num_obj - bboxes.shape[0]
        gt_dict = {
            'point_clouds': point_cloud.astype(np.float32),
            'categories': np.concatenate([categories, np.zeros((padding_num,))]).astype(np.int64),
            'centers': np.concatenate([centers, np.zeros((padding_num, 3))]).astype(np.float32),
            'heading_angles': np.concatenate([heading_angles, np.zeros((padding_num,))]).astype(np.float32),
            'heading_area_ids': np.concatenate([heading_area_ids, np.zeros((padding_num,), dtype=np.int64)]),
            'heading_offsets': np.concatenate([heading_offsets, np.zeros((padding_num,))]).astype(np.float32),
            'box_sizes': np.concatenate([box_sizes, np.zeros((padding_num, 3))]).astype(np.float32),
            'size_offsets': np.concatenate([size_offsets, np.zeros((padding_num, 3))]).astype(np.float32),
            'corners': np.concatenate([corners, np.zeros((padding_num, 8, 3))]).astype(np.float32),
            'label_masks': np.concatenate([np.ones(box_num), np.zeros(padding_num)]).astype(np.float32),
            'scan_idx': np.array(idx).astype(np.int64),
            'scan_name': scan_name
        }
        if self.return_votes:
            gt_dict['point_votes'] = point_votes.astype(np.float32)
            gt_dict['point_votes_mask'] = point_votes_mask.astype(np.int64)
        return gt_dict
