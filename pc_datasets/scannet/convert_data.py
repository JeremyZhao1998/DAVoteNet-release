import os
import argparse
import json
import csv
import numpy as np
import open3d as o3d
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def read_label_mapping(filename, label_from='raw_category', label_to='new_category'):
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            if isinstance(row, dict) and row[label_to] != '':
                mapping[row[label_from]] = row[label_to]
    return mapping


def read_mesh_vertices_rgb(filename):
    o3d_mesh = o3d.io.read_triangle_mesh(filename)
    coords = np.array(o3d_mesh.vertices)
    colors = np.array(o3d_mesh.vertex_colors)
    pc = np.hstack([coords, colors])
    return pc


def read_axis_align_matrix(filename):
    axis_align_matrix = []
    for line in open(filename).readlines():
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]
            break
    axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
    return axis_align_matrix


def random_sampling(pc, num_sample, seed=None):
    replace = (pc.shape[0] < num_sample)
    if seed is not None:
        random_state = np.random.RandomState(seed)
        choices = random_state.choice(pc.shape[0], num_sample, replace=replace)
    else:
        choices = np.random.choice(pc.shape[0], num_sample, replace=replace)
    return pc[choices], choices


def convert_data(raw_data_path, output_path, split, max_num_points=20000):
    # Read train / val split file
    idx_filename = os.path.join(BASE_DIR, 'meta_data', 'scannetv2_%s.txt' % split)
    scan_name_list = [line.rstrip() for line in open(idx_filename)]
    # Set output path
    output_folder = 'pc_bboxes_axis_aligned_' + split
    output_path = os.path.join(output_path, output_folder)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    # Read label mapping
    raw_category_dict = read_label_mapping(os.path.join(BASE_DIR, 'meta_data', 'scannet-labels.tsv'))
    obj_cnt = {name: 0 for name in raw_category_dict.values()}
    mean_sizes = {name: np.zeros(3) for name in raw_category_dict.values()}
    # Record mean color of RGB
    mean_color, scene_cnt = np.zeros(3, dtype=np.float64), 0
    for scan_name in tqdm(scan_name_list, desc='Converting data to: ' + output_folder):
        # Load mesh vertices and do calibration
        pc = read_mesh_vertices_rgb(os.path.join(raw_data_path, scan_name, scan_name + '_vh_clean.ply'))
        align_matrix = read_axis_align_matrix(os.path.join(raw_data_path, scan_name, scan_name + '.txt'))[: 3, : 3]
        pc[:, : 3] = np.dot(pc[:, : 3], align_matrix.transpose())
        # Load semantic and instance labels
        agg_data = json.load(open(os.path.join(raw_data_path, scan_name, scan_name + '.aggregation.json')))
        seg_data = json.load(open(os.path.join(raw_data_path, scan_name, scan_name + '_vh_clean.segs.json')))
        seg_points = {seg_idx: [] for seg_idx in set(seg_data['segIndices'])}
        for point_idx, seg_idx in enumerate(seg_data['segIndices']):
            seg_points[seg_idx].append(point_idx)
        obj_list, category_list = [], []
        for instance_info in agg_data['segGroups']:
            raw_category_name = instance_info['label']
            if raw_category_name not in raw_category_dict:
                continue
            category_name = raw_category_dict[raw_category_name]
            point_idx_list = []
            for seg_idx in instance_info['segments']:
                point_idx_list += seg_points[seg_idx]
            if len(point_idx_list) < 10:
                print('Empty instance: ' + instance_info['label'] + ' in ' + scan_name)
                continue
            pc_in_instance = pc[point_idx_list, :]
            # Compute instance bounding box
            x_min, x_max = np.min(pc_in_instance[:, 0]), np.max(pc_in_instance[:, 0])
            y_min, y_max = np.min(pc_in_instance[:, 1]), np.max(pc_in_instance[:, 1])
            z_min, z_max = np.min(pc_in_instance[:, 2]), np.max(pc_in_instance[:, 2])
            centroid = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2])
            l, w, h = x_max - x_min, y_max - y_min, z_max - z_min
            if l < 1e-2 or w < 1e-2 or h < 1e-2:
                print('Too little instance: ' + category_name + ' in ' + scan_name + ' with size: ', l, w, h)
                continue
            heading_angle = 0
            obj_list.append(np.array([centroid[0], centroid[1], centroid[2], l, w, h, heading_angle]))
            category_list.append(category_name)
            mean_sizes[category_name] += np.array([l, w, h])
            obj_cnt[category_name] += 1
        if len(obj_list) == 0:
            print('No object in ' + scan_name)
            continue
        bboxes = np.vstack(obj_list)
        scene_cnt += 1
        pc, choices = random_sampling(pc, max_num_points)
        mean_color += np.mean(pc[:, 3:], axis=0)
        np.savez_compressed(os.path.join(str(output_path), '%s_pc_bboxes.npz' % scan_name),
                            pc=pc, bboxes=bboxes, categories=category_list)
    # Mean color information
    mean_color /= scene_cnt
    print('Dataset Mean color: ', mean_color)
    # Mean size and object count information
    for name, cnt in obj_cnt.items():
        mean_sizes[name] /= max(1, cnt)
    for name, size in mean_sizes.items():
        print("Category: %s, Object count: %d, Mean size: %s" % (name, obj_cnt[name], size))
    # Save mean sizes
    if split == 'train':
        info_path = os.path.join(str(output_path), 'info')
        if not os.path.exists(info_path):
            os.mkdir(info_path)
        np.savez_compressed(os.path.join(str(info_path), "mean_color.npz"), mean_color=mean_color)
        np.savez_compressed(os.path.join(str(info_path), "obj_cnt.npz"), **obj_cnt)
        np.savez_compressed(os.path.join(str(info_path), "mean_sizes.npz"), **mean_sizes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_path', type=str, default='/home/zhaozj/Datasets/scannet')
    parser.add_argument('--output_root', type=str, default='/home/zhaozj/Datasets')
    parser.add_argument('--num_points', type=int, default=200000)
    args = parser.parse_args()
    for element in vars(args):
        print(element, ':', getattr(args, element))
    output = os.path.join(args.output_root, "scannet")
    if not os.path.exists(output):
        os.mkdir(output)
    convert_data(
        raw_data_path=os.path.join(args.raw_data_path, 'scans'),
        output_path=output,
        split='train',
        max_num_points=args.num_points
    )
    convert_data(
        raw_data_path=os.path.join(args.raw_data_path, 'scans'),
        output_path=output,
        split='val',
        max_num_points=args.num_points
    )
