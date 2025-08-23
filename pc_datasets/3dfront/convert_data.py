import os
import sys
import json
import random
import open3d as o3d
import argparse
import numpy as np
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../../'))
import utils.pc_utils as pc_utils
import utils.bbox_utils as bbox_utils

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


def read_mesh_vertices_rgb(filename, num_points):
    o3d_mesh = o3d.io.read_triangle_mesh(filename)
    coords = np.array(o3d_mesh.vertices)
    coords = coords[:, [0, 2, 1]] * np.array([1, -1, 1])
    colors = np.array(o3d_mesh.vertex_colors)
    pc = np.hstack([coords, colors])
    replace = (pc.shape[0] < num_points)
    choices = np.random.choice(pc.shape[0], num_points, replace=replace)
    pc = pc[choices]
    return pc


def convert_data(raw_data_path, output_path, split, axis_aligned=False, num_points=20000):
    # Set input path
    ply_file_path = os.path.join(raw_data_path, '3dfront_pc_data', split)
    anno_file_path = os.path.join(raw_data_path, '3dfront_pc_data', f'{split}_bbox_anno.json')
    anno_all = json.load(open(anno_file_path))
    room_ids = sorted(list(anno_all.keys()))
    # Set output path
    output_folder = 'pc_bboxes_axis_aligned_' + split if axis_aligned else 'pc_bboxes_' + split
    output_path = os.path.join(output_path, output_folder)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    # Record mean color of RGB
    mean_color, scene_cnt = np.zeros(3, dtype=np.float64), 0
    # Record mean sizes
    categories = ['_scene_', 'bed', 'bookshelf', 'cabinet', 'chair', 'desk', 'floor', 'lamp',
                  'night_stand', 'shelf', 'sofa', 'table', 'tv_stand', 'wardrobe']
    ignore_categories = {'_scene_', 'floor'}
    mean_sizes = {name: np.zeros(3) for name in categories if name not in ignore_categories}
    obj_cnt = {name: 0 for name in categories if name not in ignore_categories}
    for room_id in tqdm(room_ids):
        anno_data = anno_all[room_id]
        ply_file = os.path.join(str(ply_file_path), room_id + '_retrieval.ply')
        pc_np = read_mesh_vertices_rgb(ply_file, num_points)
        category_list = [categories[c] for c in anno_data['category']]
        bboxes, angles = anno_data['bbox'], anno_data['angles']
        obj_list, final_category_list = [], []
        for box_idx in range(len(category_list)):
            c, b, a = category_list[box_idx], bboxes[box_idx], angles[box_idx]
            lx, ly, lz, cx, cy, cz = b
            cy += ly / 2
            a = a[0] / 180.0 * np.pi
            if c not in ignore_categories:
                if not axis_aligned:
                    obj_list.append(np.array([cx, -cz, cy, lx, lz, ly, a]))
                    mean_sizes[c] += np.array([lx, lz, ly])
                else:
                    centers, lengths = np.array([cx, -cz, cy]), np.array([lx, lz, ly])
                    corners_3d = bbox_utils.boxes_to_corners_3d_np(centers, a, lengths)
                    _, inds = pc_utils.extract_pc_in_box3d(pc_np, corners_3d)
                    pc_in_box = pc_np[inds, :]
                    if len(pc_in_box) < 5:
                        print('Empty instance: ' + c + ' in ' + room_id)
                        continue
                    x_min, x_max = np.min(pc_in_box[:, 0]), np.max(pc_in_box[:, 0])
                    cx, lx = (x_min + x_max) / 2, x_max - x_min
                    y_min, y_max = np.min(pc_in_box[:, 1]), np.max(pc_in_box[:, 1])
                    cy, ly = (y_min + y_max) / 2, y_max - y_min
                    z_min, z_max = np.min(pc_in_box[:, 2]), np.max(pc_in_box[:, 2])
                    cz, lz = (z_min + z_max) / 2, z_max - z_min
                    obj_list.append(np.array([cx, cy, cz, lx, ly, lz, 0.0]))
                    mean_sizes[c] += np.array([lx, ly, lz])
                final_category_list.append(c)
                obj_cnt[c] += 1
        assert len(obj_list) == len(final_category_list)
        if len(obj_list) < 3 or len(obj_list) > 32:
            print('Skip scene: %s for too few or too many objects (%d objects)' % (room_id, len(obj_list)))
            continue
        bboxes = np.vstack(obj_list)
        scene_cnt += 1
        mean_color += np.mean(pc_np[:, 3:], axis=0)
        np.savez_compressed(os.path.join(str(output_path), '%s_pc_bboxes.npz' % room_id),
                            pc=pc_np, bboxes=bboxes, categories=final_category_list)
    for name, cnt in obj_cnt.items():
        mean_sizes[name] /= max(1, cnt)
    for name, size in mean_sizes.items():
        print("Category: %s, Object count: %d, Mean size: %s" % (name, obj_cnt[name], size))
    # Mean color information
    mean_color /= scene_cnt
    print('Dataset Mean color: ', mean_color)
    # Save mean sizes
    if split == 'train':
        info_path = os.path.join(str(output_path), 'info')
        if not os.path.exists(info_path):
            os.mkdir(info_path)
        np.savez_compressed(os.path.join(str(info_path), "obj_cnt.npz"), **obj_cnt)
        np.savez_compressed(os.path.join(str(info_path), "mean_sizes.npz"), **mean_sizes)
        np.savez_compressed(os.path.join(str(info_path), "mean_color.npz"), mean_color=mean_color)
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_path', type=str, default='<your_data_root>/3dfront')
    parser.add_argument('--output_root', type=str, default='<your_data_root>/3dfront')
    parser.add_argument('--num_points', type=int, default=100000)
    parser.add_argument('--axis_aligned', type=int, default=0, help='Use axis aligned boxes.')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    for element in vars(args):
        print(element, ':', getattr(args, element))
    # split_trainval(args.raw_data_path)
    convert_data(
        raw_data_path=args.raw_data_path,
        output_path=args.output_root,
        split='val',
        axis_aligned=args.axis_aligned,
        num_points=args.num_points
    )
    convert_data(
        raw_data_path=args.raw_data_path,
        output_path=args.output_root,
        split='train',
        axis_aligned=args.axis_aligned,
        num_points=args.num_points
    )
