import os
import sys
import csv
import numpy as np
import scipy
import argparse
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../../'))
import utils.pc_utils as pc_utils
import utils.bbox_utils as bbox_utils


def read_label_mapping(filename, label_from='raw_category', label_to='new_category'):
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            if isinstance(row, dict) and row[label_to] != '':
                mapping[row[label_from]] = row[label_to]
    return mapping


def convert_data(data_root, raw_data_path, split, max_num_points, v1=False, axis_aligned=False):
    idx_filename = os.path.join(raw_data_path, '%s_data_idx.txt' % split)
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]
    output_folder = 'pc_bboxes_' + ('axis_aligned_' if axis_aligned else '') + split
    output_path = os.path.join(data_root, output_folder)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    # Record mean sizes
    raw_category_dict = read_label_mapping(os.path.join(BASE_DIR, 'meta_data', 'sunrgbd-labels.tsv'))
    mean_sizes = {name: np.zeros(3) for name in raw_category_dict.values()}
    obj_cnt = {name: 0 for name in raw_category_dict.values()}
    # Record mean color
    mean_color, scene_cnt = np.zeros(3, dtype=np.float64), 0
    for data_idx in tqdm(data_idx_list, desc='Converting data to: ' + output_folder):
        # Point clouds
        depth_filename = os.path.join(raw_data_path, 'depth', '%06d.mat' % data_idx)
        pc = scipy.io.loadmat(depth_filename)['instance']
        # Read labels
        label_filename = os.path.join(raw_data_path, 'label_v1' if v1 else 'label_v2', '%06d.txt' % data_idx)
        obj_lines = [line.rstrip() for line in open(label_filename)]
        box_data = []
        for obj_line in obj_lines:
            data = obj_line.split(' ')
            data[1:] = [float(x) for x in data[1:]]
            if data[0] in raw_category_dict:
                box_data.append(data)
        # If converting axis aligned boxes, first compute the rotation angle
        # We rotate the point clouds by the weighted average angles of all objects
        # Weights are the xy areas of the boxes
        pc_aligned = np.copy(pc)
        if axis_aligned:
            angle_final, area_max = 0.0, 0.0
            for data in box_data:
                angle = np.arctan2(data[12], data[11])
                area = float(data[8]) * float(data[9])
                if area > area_max:
                    angle_final = angle if angle >= 0 else angle + np.pi
                    area_max = area
            pc_aligned[:, : 3] = np.dot(pc[:, : 3], np.transpose(pc_utils.rot_z(-angle_final)))
        # Convert boxes
        obj_list, category_list = [], []
        for data in box_data:
            category_name = raw_category_dict[data[0]]
            # Bounding box information
            centroid = np.array([data[5], data[6], data[7]])
            w, l, h = data[8] * 2, data[9] * 2, data[10] * 2
            heading_angle = np.arctan2(data[12], data[11])
            # Extract points in the box
            corners_3d = bbox_utils.boxes_to_corners_3d_np(centroid, heading_angle, np.array([l, w, h]))
            pc_in_box, inds = pc_utils.extract_pc_in_box3d(pc, corners_3d)
            pc_in_box_aligned = pc_aligned[inds, :]
            if len(pc_in_box) < 10:
                print('Empty instance: ' + category_name + ' in ' + str(data_idx))
                continue
            # Convert box to axis aligned box
            if axis_aligned:
                x_min, x_max = np.min(pc_in_box_aligned[:, 0]), np.max(pc_in_box_aligned[:, 0])
                y_min, y_max = np.min(pc_in_box_aligned[:, 1]), np.max(pc_in_box_aligned[:, 1])
                centroid = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, centroid[2]])
                l, w = x_max - x_min, y_max - y_min
                heading_angle = 0
            if l < 1e-2 or w < 1e-2 or float(h) < 1e-2:
                print('Too little instance: ' + category_name + ' in ' + str(data_idx) + ' with size: ', l, w, h)
                continue
            mean_sizes[category_name] += np.array([l, w, h])
            obj_cnt[category_name] += 1
            obj_list.append(np.array([centroid[0], centroid[1], centroid[2], l, w, h, heading_angle]))
            category_list.append(category_name)
        if len(obj_list) == 0:
            print('No object in ' + str(data_idx))
            continue
        bboxes = np.vstack(obj_list)
        pc, choices = pc_utils.random_sampling(pc, max_num_points, return_choices=True)
        mean_color += np.mean(pc[:, 3:], axis=0)
        scene_cnt += 1
        pc_aligned = pc_aligned[choices, :]
        np.savez_compressed(os.path.join(str(output_path), '%06d_pc_bboxes.npz' % data_idx),
                            pc=(pc_aligned if axis_aligned else pc), bboxes=bboxes, categories=category_list)
    # Mean color information
    mean_color /= scene_cnt
    print('Dataset Mean color: ', mean_color)
    for name, cnt in obj_cnt.items():
        mean_sizes[name] /= max(1, cnt)
    for name, size in mean_sizes.items():
        print("Category: %s, Object count: %d, Mean size: %s" % (name, obj_cnt[name], size))
    # Save mean sizes
    if split == 'train':
        info_path = os.path.join(str(output_path), 'info')
        if not os.path.exists(info_path):
            os.mkdir(info_path)
        np.savez_compressed(os.path.join(str(info_path), "mean_sizes.npz"), **mean_sizes)
        np.savez_compressed(os.path.join(str(info_path), "obj_cnt.npz"), **obj_cnt)
        np.savez_compressed(os.path.join(str(info_path), "mean_color.npz"), mean_color=mean_color)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='<your_data_root>/sunrgbd')
    parser.add_argument('--raw_data_path', type=str, default='sunrgbd_trainval')
    parser.add_argument('--num_points', type=int, default=200000)
    parser.add_argument('--v1', type=int, default=0, help='Use SUN_RGBD v1 data split.')
    parser.add_argument('--axis_aligned', type=int, default=1, help='Use axis aligned boxes.')
    args = parser.parse_args()
    args.v1 = bool(args.v1)
    args.axis_aligned = bool(args.axis_aligned)
    for element in vars(args):
        print(element, ':', getattr(args, element))
    # Random seed
    np.random.seed(0)
    convert_data(
        data_root=args.data_root,
        raw_data_path=os.path.join(args.data_root, args.raw_data_path),
        split='train',
        max_num_points=args.num_points,
        v1=args.v1,
        axis_aligned=args.axis_aligned
    )
    convert_data(
        data_root=args.data_root,
        raw_data_path=os.path.join(args.data_root, args.raw_data_path),
        split='val',
        max_num_points=args.num_points,
        v1=args.v1,
        axis_aligned=args.axis_aligned
    )
