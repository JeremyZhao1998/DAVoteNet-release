import os
import argparse
import numpy as np
from tqdm import tqdm

from pc_datasets import PCDetectionDataset
import utils.pc_utils as pc_utils
import utils.bbox_utils as bbox_utils
import pc_datasets.augmentations as aug


def parse_args():
    parser = argparse.ArgumentParser()
    # Dataset configuration
    parser.add_argument('--data_root', default='/home/zhaozj/Datasets', help='Data root')
    parser.add_argument('--src_dataset', default='procthor', help='Dataset name')
    parser.add_argument('--split_set', default='train', help='Split set')
    parser.add_argument('--axis_aligned', type=int, default=1, help='Use axis aligned bounding boxes')
    parser.add_argument('--noise_jitter', type=float, default=0.01, help='Noise jitter')
    parser.add_argument('--view', type=int, default=4, help='Occlusion views')
    parser.add_argument('--radius', type=int, default=1000, help='Occlusion radius')
    parsed_args = parser.parse_args()
    return parsed_args


def main():
    src_dataset = PCDetectionDataset(
        data_root=args.data_root,
        dataset_name=args.src_dataset,
        categories=['chair', 'table'],
        split_set=args.split_set,
        axis_aligned=args.axis_aligned
    )
    data_folder = 'pc_bboxes_' + ('axis_aligned_' if args.axis_aligned else '') + args.split_set
    output_path = os.path.join(args.data_root, args.src_dataset + '_vss', data_folder)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if len(list(os.listdir(str(output_path)))) >= len(src_dataset):
        print('VSS augmented data of %s already exists.' % (args.src_dataset + '_' + args.split_set))
        return
    for idx in tqdm(range(len(src_dataset)), desc='Processing VSS data'):
        scan_name = src_dataset.scan_names[idx]
        data_dict = np.load(os.path.join(str(src_dataset.data_path), scan_name) + '_pc_bboxes.npz')
        point_cloud, bboxes, category_names = data_dict['pc'], data_dict['bboxes'], data_dict['categories']
        src_dataset.num_points = point_cloud.shape[0]
        categories = np.array([1 for _ in range(bboxes.shape[0])])
        centers = bboxes[:, : 3]
        box_sizes = bboxes[:, 3: 6]
        heading_angles = bboxes[:, 6] if not args.axis_aligned else np.zeros((bboxes.shape[0],))
        corners = np.stack([bbox_utils.boxes_to_corners_3d_np(c, a, s)
                            for c, a, s in zip(centers, heading_angles, box_sizes)], axis=0) \
            if len(bboxes) > 0 else np.zeros((0, 8, 3))
        _, _, label_mask = src_dataset.get_point_votes(point_cloud, centers, corners, categories)
        new_pc = aug.virtual_scan_simulation(point_cloud, label_mask, args.noise_jitter, args.view, args.radius)
        new_pc = pc_utils.random_sampling(new_pc, len(point_cloud))
        np.savez_compressed(os.path.join(str(output_path), '%s_pc_bboxes.npz' % scan_name),
                            pc=new_pc, bboxes=bboxes, categories=category_names)
    # copy other info files of original dataset
    if args.split_set == 'train':
        os.system('cp -r %s %s/.' % (str(src_dataset.info_path), output_path))
    pass


if __name__ == '__main__':
    args = parse_args()
    main()
