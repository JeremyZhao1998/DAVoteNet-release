import os
import sys
import csv
import gzip
import json
import random
import argparse
import trimesh
import numpy as np
from tqdm import tqdm
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../../'))
import utils.pc_utils as pc_utils


def read_label_mapping(filename, label_from='raw_category', label_to='new_category'):
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            if isinstance(row, dict) and row[label_to] != '':
                mapping[row[label_from]] = row[label_to]
    return mapping


def srgb_to_linear(image: Image.Image) -> Image.Image:
    """
    Convert an sRGB image to linear color space.
    This is used for baseColorTexture correction in GLTF.
    """
    # Ensure image is in RGB mode
    image = image.convert("RGB")

    # Normalize to 0-1
    arr = np.asarray(image).astype(np.float32) / 255.0

    # Apply inverse gamma correction
    def srgb_channel_to_linear(c):
        return np.where(c <= 0.04045,
                        c / 12.92,
                        ((c + 0.055) / 1.055) ** 2.4)

    linear_arr = srgb_channel_to_linear(arr)

    # Convert back to 0-255 uint8
    linear_img = Image.fromarray((np.clip(linear_arr, 0.0, 1.0) * 255).astype(np.uint8), mode="RGB")
    return linear_img


def sample_points_from_mesh(scene_path, additional_path, num_points):
    os.system(f"mkdir -p {scene_path}")
    os.system(f"tar -xf {scene_path}.tar.xz -C {scene_path}")
    os.system(f"tar -xf {additional_path}.tar.xz -C {additional_path}")
    scene = trimesh.Scene()
    obj_list = [file for file in os.listdir(scene_path) if file.endswith('.obj')]
    for file in obj_list:
        file_name = file.split('.')[0]
        if file_name in os.listdir(additional_path):
            obj = trimesh.load_mesh(os.path.join(additional_path, file_name, 'raw_model.obj'))
            scene += obj
        else:
            obj = trimesh.load_mesh(os.path.join(scene_path, file))
            scene += obj
    points_cnt, mesh_list, points_all, colors_all = [], [], [], []
    for mesh in scene.geometry.values():
        mesh_list.append(mesh)
        name = mesh.metadata["name"] if "name" in mesh.metadata else ""
        if name.startswith("wall") or name.startswith("ceiling"):
            points_cnt.append(mesh.area * 0.1)
        elif name.startswith("room"):
            points_cnt.append(mesh.area * 0.5)
        else:
            points_cnt.append(mesh.area)
    points_cnt = np.round(np.array(points_cnt) / np.sum(points_cnt) * num_points).astype(np.int64)
    for mesh, cnt in zip(mesh_list, points_cnt):
        if cnt == 0:
            continue
        try:
            if hasattr(mesh.visual, 'material') and hasattr(mesh.visual.material, 'to_simple'):
                mesh.visual.material = mesh.visual.material.to_simple()
            if mesh.visual.uv is None:
                points, face_idx = trimesh.sample.sample_surface(mesh, count=cnt, sample_color=False)
                colors = None
            else:
                points, face_idx, colors = trimesh.sample.sample_surface(mesh, count=cnt, sample_color=True)
            if colors is None:
                colors = np.vstack([mesh.visual.material.main_color for _ in range(cnt)])
            colors = colors / 255.0
            points_all.append(points)
            colors_all.append(colors)
        except Exception as e:
            print(e)
            continue
    if len(points_all) == 0:
        print('No points sampled from mesh: %s' % scene_path)
        return None
    points_all = np.concatenate(points_all, axis=0) * np.array([1, 1, 1])
    colors_all = np.concatenate(colors_all, axis=0)
    pc_np = np.hstack([points_all, colors_all])
    os.system(f"rm -r {scene_path}")
    os.system(f"rm -r {additional_path}")
    return pc_np


def convert_data(raw_data_path, front_data_path, output_path, split, axis_aligned, num_points=200000):
    mesh_root = os.path.join(raw_data_path, f'{split}_mesh')
    front_root = os.path.join(front_data_path, f'{split}_mesh') if front_data_path is not None else None
    anno_root = os.path.join(raw_data_path, f'{split}_anno')
    scene_names = sorted([name.split('.')[0] for name in os.listdir(mesh_root)])
    # Set output path
    output_folder = 'pc_bboxes_' + ('axis_aligned_' if axis_aligned else '') + split
    output_path = os.path.join(output_path, output_folder)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    # Record mean color of RGB
    mean_color, scene_cnt = np.zeros(3, dtype=np.float64), 0
    # Record mean sizes
    raw_category_dict = read_label_mapping(os.path.join(BASE_DIR, 'meta_data', 'simroom-labels.tsv'))
    mean_sizes = {name: np.zeros(3) for name in raw_category_dict.values()}
    obj_cnt = {name: 0 for name in raw_category_dict.values()}
    for scene_name in tqdm(scene_names, desc='Converting data to: ' + output_folder):
        anno_file = os.path.join(anno_root, scene_name + '.json.gz')
        with gzip.open(anno_file, 'rb') as f:
            anno_info = json.load(f)
        front_dir = os.path.join(front_root, scene_name) if front_root is not None else None
        pc_room = sample_points_from_mesh(os.path.join(mesh_root, scene_name), front_dir, num_points)
        if pc_room is None:
            continue
        obj_list, category_list = [], []
        for instance_info in anno_info:
            raw_category_name = instance_info['objectType'].lower()
            if raw_category_name not in raw_category_dict:
                continue
            category_name = raw_category_dict[raw_category_name]
            if instance_info['objectOrientedBoundingBox'] is not None and not axis_aligned:
                corners = np.array(instance_info['objectOrientedBoundingBox']['cornerPoints'])
                corners = np.array(corners)[:, [0, 2, 1]] * np.array([-1, -1, 1])
                try:
                    _, inds = pc_utils.extract_pc_in_box3d(pc_room, corners)
                except ValueError:
                    continue
                pc_instance = pc_room[inds, :3]
                angle = instance_info['rotation']['y'] / 180.0 * np.pi
                pc_instance = pc_instance @ pc_utils.rot_z(-angle)
                if len(pc_instance) < 5:
                    # print('Empty instance: ' + category_name + ' in ' + scene_name)
                    continue
                min_coords, max_coords = np.mean(pc_instance, axis=0), np.max(pc_instance, axis=0)
                center = (max_coords + min_coords) / 2
                size = max_coords - min_coords
                cx, cy, cz = center[0], center[1], center[2]
                lx, ly, lz = size[0], size[1], size[2]
            else:
                corners = np.array(instance_info['axisAlignedBoundingBox']['cornerPoints'])
                if corners is None or len(corners.shape) == 0:
                    continue
                corners = np.array(corners)[:, [0, 2, 1]] * np.array([-1, -1, 1])
                try:
                    _, inds = pc_utils.extract_pc_in_box3d(pc_room, corners)
                except ValueError:
                    continue
                pc_instance = pc_room[inds, :3]
                if len(pc_instance) < 5:
                    # print('Empty instance: ' + category_name + ' in ' + scene_name)
                    continue
                center = instance_info['axisAlignedBoundingBox']['center']
                cx, cy, cz = -center['x'], -center['z'], center['y']
                size = instance_info['axisAlignedBoundingBox']['size']
                lx, ly, lz = size['x'], size['z'], size['y']
                angle = 0.0
            if lx < 1e-2 or ly < 1e-2 or lz < 1e-2:
                continue
            obj_list.append(np.array([cx, cy, cz, lx, ly, lz, angle]))
            category_list.append(category_name)
            mean_sizes[category_name] += np.array([lx, ly, lz])
            obj_cnt[category_name] += 1
        pc_room, choices = pc_utils.random_sampling(pc_room, num_points, return_choices=True)
        mean_color += np.mean(pc_room[:, 3: 6], axis=0)
        assert len(obj_list) == len(category_list)
        if len(obj_list) < 3:
            print('Too few objects in scene: %s' % scene_name)
            continue
        scene_cnt += 1
        bboxes = np.vstack(obj_list)
        np.savez_compressed(os.path.join(str(output_path), '%s_pc_bboxes.npz' % scene_name),
                            pc=pc_room, bboxes=bboxes, categories=category_list)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_path', type=str, default='<your_data_root>/procthor')
    parser.add_argument('--front_data_path', type=str, default=None)
    parser.add_argument('--output_root', type=str, default='<your_data_root>/procthor')
    parser.add_argument('--axis_aligned', type=int, default=1, help='Use axis aligned boxes.')
    parser.add_argument('--num_points', type=int, default=200000)
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    args = parser.parse_args()
    for element in vars(args):
        print(element, ':', getattr(args, element))
    seed = args.seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    if not os.path.exists(args.output_root):
        os.mkdir(args.output_root)
    convert_data(
        raw_data_path=args.raw_data_path,
        front_data_path=args.front_data_path,
        output_path=args.output_root,
        split='train',
        axis_aligned=args.axis_aligned,
        num_points=args.num_points
    )
    convert_data(
        raw_data_path=args.raw_data_path,
        front_data_path=args.front_data_path,
        output_path=args.output_root,
        split='val',
        axis_aligned=args.axis_aligned,
        num_points=args.num_points
    )
