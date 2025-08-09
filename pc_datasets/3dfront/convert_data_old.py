import os
import json
import random
import trimesh
import argparse
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def read_label_mapping(filename, label_from='raw_category', label_to='new_category'):
    import csv
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            if isinstance(row, dict) and row[label_to] != '':
                mapping[row[label_from]] = row[label_to]
    return mapping


def split_trainval(raw_data_path, train_num=20000):
    anno_data_path = os.path.join(raw_data_path, '3D-FRONT')
    all_scene_list = os.listdir(anno_data_path)
    all_scene_list = [line.split('.')[0] for line in all_scene_list]
    all_room_list = []
    for scene_id in tqdm(all_scene_list):
        anno_data = json.load(open(os.path.join(anno_data_path, scene_id + '.json')))
        for room in anno_data['scene']['room']:
            if room['empty'] == 0:
                all_room_list.append(scene_id + '_' + room['instanceid'])
    random.shuffle(all_room_list)
    train_list = all_room_list[:train_num]
    val_list = list(set(all_room_list) - set(train_list))
    train_list, val_list = sorted(train_list), sorted(val_list)
    train_list_file = os.path.join(BASE_DIR, 'meta_data', 'train_list.txt')
    val_list_file = os.path.join(BASE_DIR, 'meta_data', 'val_list.txt')
    with open(train_list_file, 'w') as f:
        for line in train_list:
            f.write(line + '\n')
    with open(val_list_file, 'w') as f:
        for line in val_list:
            f.write(line + '\n')


def extract_categories(raw_data_path):
    model_meta_path = os.path.join(raw_data_path, '3D-FUTURE-model', 'model_info.json')
    model_meta = json.load(open(model_meta_path))
    id_category_dict = {data['model_id']: None for data in model_meta}
    for data in model_meta:
        model_id = data['model_id']
        if data['category'] is not None:
            id_category_dict[model_id] = data['category'] + ' - ' + data['super-category']
        elif data['super-category'] is not None:
            id_category_dict[model_id] = data['super-category']
    return id_category_dict


def adjust_obj(obj, pos, rot, scale, axis_aligned=False, restrict_size=None, check_size=True, s_bounds=None):
    obj.apply_scale(np.array(scale))
    ori_bounds = np.array(obj.bounds)
    ori_size = ori_bounds[1] - ori_bounds[0]
    if restrict_size is not None:
        new_scale = restrict_size / ori_size
        obj.apply_scale(new_scale)
        ori_bounds = np.array(obj.bounds)
        ori_size = ori_bounds[1] - ori_bounds[0]
    rotation = R.from_quat(rot)
    r_matrix = np.eye(4)
    r_matrix[:3, :3] = rotation.as_matrix()
    obj.apply_transform(r_matrix)
    obj.apply_translation(np.array(pos))
    bounds = np.array(obj.bounds)
    bound_centers = np.mean(bounds, axis=0)
    bound_size = bounds[1] - bounds[0]
    if axis_aligned or not np.isclose(bound_size[1], ori_size[1]):
        centers, size, angle = bound_centers, bound_size, 0.0
    else:
        euler_z = rotation.as_euler('xyz')[1]
        ori_centers = np.mean(ori_bounds, axis=0)
        centers, size, angle = ori_centers + np.array(pos), ori_size, euler_z
    if check_size:
        if np.isnan(size).any() or np.isnan(centers).any() or (size < 1e-2).any():
            return None, None, None, None
    if s_bounds is not None and ((bounds[0] < s_bounds[0]).any() or (bounds[1] > s_bounds[1]).any()):
        return None, None, None, None
    return obj, centers, size, angle


def export_ply(pc_np):
    N = pc_np.shape[0]
    header = f'''ply
            format ascii 1.0
            element vertex {N}
            property float x
            property float y
            property float z
            end_header
            '''
    with open('test.ply', 'w') as f:
        f.write(header)
        for p in pc_np:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")


def convert_data(raw_data_path, output_path, split, axis_aligned=False, num_points=20000):
    anno_data_path = os.path.join(raw_data_path, '3D-FRONT')
    model_data_path = os.path.join(raw_data_path, '3D-FUTURE-model')
    category_info = extract_categories(raw_data_path)
    valid_model_ids = set(os.listdir(model_data_path))
    texture_data_path = os.path.join(raw_data_path, '3D-FRONT-texture')
    valid_texture_ids = set(os.listdir(texture_data_path))
    split_file = os.path.join(BASE_DIR, 'meta_data', split + '_list.txt')
    room_ids = [line.rstrip() for line in open(split_file)]
    # Set output path
    output_folder = 'pc_bboxes_axis_aligned_' + split if axis_aligned else 'pc_bboxes_' + split
    output_path = os.path.join(output_path, output_folder)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    # Record mean color of RGB
    mean_color, scene_cnt = np.zeros(3, dtype=np.float64), 0
    # Record mean sizes
    raw_category_dict = read_label_mapping(os.path.join(BASE_DIR, 'meta_data', '3dfront-labels.tsv'))
    new_category_list = [name for name in raw_category_dict.values() if name is not None]
    mean_sizes = {name: np.zeros(3) for name in new_category_list}
    obj_cnt = {name: 0 for name in new_category_list}
    for room_id in tqdm(room_ids):
        scene_id, room_instanceid = room_id.split('_')
        anno_data = json.load(open(os.path.join(anno_data_path, scene_id + '.json')))
        # furniture source data
        furniture_data = [fur for fur in anno_data['furniture'] if fur['jid'] in valid_model_ids]
        furniture_data = {fur['uid']: fur for fur in furniture_data}
        # material source data for mesh texture
        material_data = anno_data['material']
        material_data = {ma['uid']: ma for ma in material_data}
        # mesh source data of walls, floors, ceilings and windows
        mesh_data = anno_data['mesh']
        mesh_data = {me['uid']: me for me in mesh_data}
        # Element data in rooms
        furniture_list, mesh_list = [], []
        for room in anno_data['scene']['room']:
            if room['empty'] == 0 and room['instanceid'] == room_instanceid:
                for ele in room['children']:
                    if ele['ref'] in furniture_data:
                        furniture_list.append(ele)
                    elif ele['ref'] in mesh_data:
                        mesh_list.append(ele)
        if len(furniture_list) == 0:
            print('Skip scene: %s for no objects' % scene_id)
            continue
        # Trimesh scene
        scene = trimesh.Scene()
        sample_factor = []
        obj_list, category_list = [], []
        # Process mesh objects
        for ele in mesh_list:
            me = mesh_data[ele['ref']]
            material = material_data[me['material']]
            vertices, faces = np.array(me['xyz']).reshape(-1, 3), np.array(me['faces']).reshape(-1, 3)
            vertex_normals = np.array(me['normal']).reshape(-1, 3)
            if vertices.shape != vertex_normals.shape or faces.shape[0] == 0 or vertices.shape[0] == 0:
                continue
            if material['jid'] in valid_texture_ids:
                texture_file_name = os.listdir(os.path.join(texture_data_path, material['jid']))[0]
                image = Image.open(os.path.join(texture_data_path, material['jid'], texture_file_name))
                uv = np.array(me['uv']).reshape(-1, 2)
                visual = trimesh.base.TextureVisuals(uv=uv, image=image)
                obj = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=vertex_normals, visual=visual)
            else:
                color = np.array(material['color'])
                face_colors = np.repeat(color.reshape(1, 4), len(faces), axis=0)
                obj = trimesh.Trimesh(vertices=vertices, faces=faces,
                                      vertex_normals=vertex_normals, face_colors=face_colors)
            pos, rot, scale = ele['pos'], ele['rot'], ele['scale']
            if 'floor' in me['type'].lower():
                obj, centers, size, angle = adjust_obj(obj, pos, rot, scale, axis_aligned, check_size=False)
                if obj is not None:
                    scene += obj
                    sample_factor.append(0.5)
        # Process furniture objects
        for ele in furniture_list:
            fur = furniture_data[ele['ref']]
            model_id = fur['jid']
            raw_category = category_info[model_id]
            title = fur['title'] if 'title' in fur else None
            if raw_category is None or raw_category == 'Others':
                if title is not None and title != '':
                    raw_category = title
            try:
                obj = trimesh.load_mesh(os.path.join(model_data_path, model_id, 'raw_model.obj'))
            except Exception as e:
                print('Error loading model %s: %s' % (model_id, e))
                continue
            pos, rot, scale = ele['pos'], ele['rot'], ele['scale']
            restrict_size = fur['bbox'] if len(fur['bbox']) == 3 else fur['bbox'][0]
            ori_size = np.array([restrict_size[0], restrict_size[2], restrict_size[1]])
            ori_size = None
            obj, centers, size, angle = adjust_obj(obj, pos, rot, scale, axis_aligned, restrict_size=ori_size)
            if obj is None:
                continue
            scene += obj
            sample_factor.append(1.0)
            if raw_category not in raw_category_dict or raw_category_dict[raw_category] is None:
                continue
            obj_list.append(np.array([centers[0], -centers[2], centers[1], size[0], size[2], size[1], angle]))
            category_list.append(raw_category_dict[raw_category])
            mean_sizes[raw_category_dict[raw_category]] += size
            obj_cnt[raw_category_dict[raw_category]] += 1
        areas = np.array([mesh.area * factor for mesh, factor in zip(scene.geometry.values(), sample_factor)])
        areas = np.nan_to_num(areas)
        points_cnt = np.round(areas / np.sum(areas) * num_points).astype(np.int64)
        points_all, colors_all = [], []
        for mesh, cnt in zip(scene.geometry.values(), points_cnt):
            try:
                points, face_idx, colors = trimesh.sample.sample_surface(mesh, count=cnt, sample_color=True)
            except OSError:
                points, face_idx = trimesh.sample.sample_surface(mesh, count=cnt, sample_color=False)
                colors = None
            colors = np.ones((len(points), 4)) if colors is None else colors
            colors = colors / 255.0
            points_all.append(points)
            colors_all.append(colors)
        points_all = np.concatenate(points_all, axis=0)[:, [0, 2, 1]] * np.array([1, -1, 1])
        colors_all = np.concatenate(colors_all, axis=0)
        mean_color += np.mean(colors_all, axis=0)[:3]
        pc_np = np.hstack([points_all, colors_all])[:, :6]
        """ori_np = np.load('/home/zhaozj/Downloads/density1250/04684207-3d45-4d33-bd9d-0f66c9a45402_MasterBedroom-9404.npy')
        from utils.visualization import draw_point_cloud
        draw_point_cloud(pc_np)
        # draw_point_cloud(ori_np[:, :3])
        # export_ply(pc_np)
        print(room_id)
        print(category_list)
        exit()"""
        assert len(obj_list) == len(category_list)
        if len(obj_list) < 1 or len(obj_list) > 32:
            print('Skip scene: %s for too few or too many objects (%d objects)' % (scene_id, len(obj_list)))
            continue
        bboxes = np.vstack(obj_list)
        scene_cnt += 1
        np.savez_compressed(os.path.join(str(output_path), '%s_pc_bboxes.npz' % scene_id),
                            pc=pc_np, bboxes=bboxes, categories=category_list)
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
    parser.add_argument('--raw_data_path', type=str, default='/home/zhaozj/Datasets/3dfront')
    parser.add_argument('--output_root', type=str, default='/home/zhaozj/Datasets/3dfront')
    parser.add_argument('--num_points', type=int, default=200000)
    parser.add_argument('--axis_aligned', type=int, default=1, help='Use axis aligned boxes.')
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
        split='train',
        axis_aligned=args.axis_aligned,
        num_points=args.num_points
    )
    convert_data(
        raw_data_path=args.raw_data_path,
        output_path=args.output_root,
        split='val',
        axis_aligned=args.axis_aligned,
        num_points=args.num_points
    )
