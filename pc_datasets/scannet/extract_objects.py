# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Updated by Zijing Zhao (Peking University)
# Date: Oct 2024

import os
import sys
import argparse
import json
import numpy as np
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../../'))
from convert_data import read_label_mapping, read_mesh_vertices_rgb, read_axis_align_matrix


def extract_objects(raw_data_path, output_path, split):
    # Read train / val split file
    idx_filename = os.path.join(BASE_DIR, 'meta_data', 'scannetv2_%s.txt' % split)
    scan_name_list = [line.rstrip() for line in open(idx_filename)]
    # Set output path
    output_folder = 'extracted_objects_' + split
    output_path = os.path.join(output_path, output_folder)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    # Read label mapping
    raw_category_dict = read_label_mapping(os.path.join(BASE_DIR, 'meta_data', 'scannet-labels.tsv'))
    obj_cnt = {name: 0 for name in raw_category_dict.values()}
    for scan_name in tqdm(scan_name_list, desc='Extracting objects to: ' + output_folder):
        # Load mesh vertices and do calibration
        pc = read_mesh_vertices_rgb(os.path.join(raw_data_path, scan_name, scan_name + '_vh_clean.ply'))
        align_matrix = read_axis_align_matrix(os.path.join(raw_data_path, scan_name, scan_name + '.txt'))[: 3, : 3]
        pc[:, : 3] = np.dot(pc[:, : 3], align_matrix.transpose())
        # Load semantic and instance labels
        agg_data = json.load(open(os.path.join(raw_data_path, scan_name, scan_name + '_vh_clean.aggregation.json')))
        seg_data = json.load(open(os.path.join(raw_data_path, scan_name, scan_name + '_vh_clean.segs.json')))
        seg_points = {seg_idx: [] for seg_idx in set(seg_data['segIndices'])}
        for point_idx, seg_idx in enumerate(seg_data['segIndices']):
            seg_points[seg_idx].append(point_idx)
        for instance_info in agg_data['segGroups']:
            raw_category_name = instance_info['label']
            if raw_category_name not in raw_category_dict:
                continue
            category_name = raw_category_dict[raw_category_name]
            point_idx_list = []
            for seg_idx in instance_info['segments']:
                point_idx_list += seg_points[seg_idx]
            if len(point_idx_list) < 10:
                continue
            pc_in_instance = pc[point_idx_list, :]
            # Compute instance bounding box
            size = np.max(pc_in_instance[:, : 3], axis=0) - np.min(pc_in_instance[:, : 3], axis=0)
            l, w, h = size[0], size[1], size[2]
            center = (np.max(pc_in_instance[:, : 3], axis=0) + np.min(pc_in_instance[:, : 3], axis=0)) / 2
            pc_in_instance[:, : 3] -= center
            if l < 1e-2 or w < 1e-2 or h < 1e-2:
                continue
            obj_name = '%s_%s_%d_%.4f_%.4f_%.4f.npy' % (scan_name, category_name, obj_cnt[category_name], l, w, h)
            np.save(str(os.path.join(str(output_path), obj_name)), pc_in_instance)
            obj_cnt[category_name] += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_path', type=str, default='/network_space/storage43/shared_dataset/scannet')
    parser.add_argument('--output_root', type=str, default='/network_space/storage43/zhaozijing/datasets')
    args = parser.parse_args()
    for element in vars(args):
        print(element, ':', getattr(args, element))
    output = os.path.join(args.output_root, "scannet")
    if not os.path.exists(output):
        os.mkdir(output)
    extract_objects(
        raw_data_path=os.path.join(args.raw_data_path, 'scans'),
        output_path=output,
        split='train',
    )
    extract_objects(
        raw_data_path=os.path.join(args.raw_data_path, 'scans'),
        output_path=output,
        split='val',
    )
